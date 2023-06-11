#include <fstream>
#include <iostream>
#include <string>

#include <tao/pegtl/contrib/parse_tree.hpp>
#include <tao/pegtl/contrib/parse_tree_to_dot.hpp>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "Calc/CalcDialect.h"
#include "Calc/CalcOps.h"
#include "grammar.h"

// source:
// https://github.com/taocpp/PEGTL/blob/c5eed0d84b5420dc820a7c0a381a87d68f5a1038/src/example/pegtl/parse_tree.cpp#L42
struct Rearrange : tao::pegtl::parse_tree::apply<Rearrange> {
  // recursively rearrange nodes. the basic principle is:
  //
  // from:          TERM/EXPR
  //                /   |   \          (LHS... may be one or more children,
  //                followed by OP,)
  //             LHS... OP   RHS       (which is one operator, and RHS, which is
  //             a single child)
  //
  // to:               OP
  //                  /  \             (OP now has two children, the original
  //                  TERM/EXPR and RHS)
  //         TERM/EXPR    RHS          (Note that TERM/EXPR has two fewer
  //         children now)
  //             |
  //            LHS...
  //
  // if only one child is left for LHS..., replace the TERM/EXPR with the child
  // directly. otherwise, perform the above transformation, then apply it
  // recursively until LHS... becomes a single child, which then replaces the
  // parent node and the recursion ends.
  template <typename Node, typename... States>
  static void transform(std::unique_ptr<Node> &node, States &&...states) {
    if (node->children.size() == 1) {
      node = std::move(node->children.back());
    } else {
      // node is term or expression
      node->remove_content();
      auto &vec = node->children;

      // pop right hand side from children
      auto rhs = std::move(vec.back());
      vec.pop_back();

      // pop operator node from children
      auto op = std::move(vec.back());
      vec.pop_back();

      // add TERM/EXPR node as first child on the operator node
      op->children.emplace_back(std::move(node));

      // add operator node as second child on the operator node
      op->children.emplace_back(std::move(rhs));

      // replace current node with op node, so it is at the top of the subtree
      node = std::move(op);

      // recursively apply algorithm on LHS
      transform(node->children.front(), states...);
    }
  }
};

template <typename Rule>
using parseTreeSelector = tao::pegtl::parse_tree::selector<
    Rule,
    tao::pegtl::parse_tree::store_content::on<
        // clang-format off
        grammar::integer,
        grammar::identifier,
        grammar::assignment,
        grammar::add_op,
        grammar::mul_op
        // clang-format on
        >,
    Rearrange::on<
        // clang-format off
        grammar::term,
        grammar::expression
        // clang-format on
        >>;

using AstNode = std::unique_ptr<tao::pegtl::parse_tree::node>;

class MLIRGen {
  mlir::OpBuilder builder;
  mlir::Value generateMLIR(const AstNode &node);
  mlir::StringAttr filename;

public:
  MLIRGen(mlir::MLIRContext &context, mlir::ModuleOp &module,
          const std::string &filename)
      : builder(&context) {
    builder.setInsertionPointToStart(module.getBody());
    this->filename = builder.getStringAttr(filename);
  }
  void generateRoot(const AstNode &node);
};

mlir::Value MLIRGen::generateMLIR(const AstNode &node) {
  auto location = mlir::FileLineColLoc::get(filename, node->begin().line,
                                            node->begin().column);
  // integer
  if (node->is_type<grammar::integer>()) {
    int64_t value = atoi(node->string().c_str());
    return builder.create<calc::ConstantOp>(location, value);
  }

  // identifier
  if (node->is_type<grammar::identifier>()) {
    auto varName = builder.getStringAttr(node->string());
    return builder.create<calc::GetVariableOp>(location, varName);
  }

  // add and sub
  if (node->is_type<grammar::add_op>()) {
    auto lhs = generateMLIR(node->children[0]);
    auto rhs = generateMLIR(node->children[1]);
    if (node->string() == "+") {
      return builder.create<calc::AddOp>(location, lhs, rhs);
    } else if (node->string() == "-") {
      return builder.create<calc::SubOp>(location, lhs, rhs);
    } else {
      llvm_unreachable("unsupported add_op operator");
    }
  }

  // mul and div
  if (node->is_type<grammar::mul_op>()) {
    auto lhs = generateMLIR(node->children[0]);
    auto rhs = generateMLIR(node->children[1]);
    if (node->string() == "*") {
      return builder.create<calc::MulOp>(location, lhs, rhs);
    } else if (node->string() == "/") {
      return builder.create<calc::DivOp>(location, lhs, rhs);
    } else {
      llvm_unreachable("unsupported mul_op operator");
    }
  }

  llvm_unreachable("unhandled AstNode type");
}

void MLIRGen::generateRoot(const AstNode &node) {
  auto location = mlir::FileLineColLoc::get(filename, node->begin().line,
                                            node->begin().column);
  if (node->is_type<grammar::assignment>()) {
    assert(node->children.size() == 2);

    // identifier
    const AstNode &identifierNode = node->children[0];
    assert(identifierNode->is_type<grammar::identifier>());
    const std::string identifier = identifierNode->string();

    // value
    const AstNode &valueNode = node->children[1];
    mlir::Value value = generateMLIR(valueNode);

    builder.create<calc::SetVariableOp>(location, identifier, value);
  } else {
    mlir::Value value = generateMLIR(node);
    builder.create<calc::PrintOp>(location, value);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Error: Please provide exactly one file name as a parameter."
              << std::endl;
    return 1;
  }

  std::string fileName = argv[1];
  std::ifstream inputFile(fileName);

  // Check if the file is opened successfully
  if (!inputFile) {
    std::cerr << "Error: Unable to open the file: " << fileName << std::endl;
    return 1;
  }

  // Read the file content into an std::string
  std::stringstream fileContent;
  fileContent << inputFile.rdbuf();
  inputFile.close();

  std::string inStr = fileContent.str();
  tao::pegtl::memory_input<> in(inStr, fileName);

  AstNode root =
      tao::pegtl::parse_tree::parse<grammar::start, parseTreeSelector>(in);

  if (!root) {
    std::cerr << "parse failed" << std::endl;
    return 1;
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<calc::CalcDialect>();

  auto fileNameAttr = mlir::StringAttr::get(&context, fileName);
  auto location = mlir::FileLineColLoc::get(fileNameAttr, 0, 0);

  mlir::ModuleOp module = mlir::ModuleOp::create(location);
  MLIRGen gen(context, module, fileName);

  for (const AstNode &node : root->children) {
    gen.generateRoot(node);
  }

  module.dump();

  // Verify the function and module.
  if (mlir::failed(mlir::verify(module))) {
    llvm::errs() << "Error: module verification failed\n";
    return 1;
  }
  return 0;
}
