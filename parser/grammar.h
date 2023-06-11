#include <tao/pegtl.hpp>
#include <tao/pegtl/ascii.hpp>

namespace grammar {

using namespace tao::pegtl;

struct ws : star<blank> {};

struct integer : seq<opt<one<'-'>>, plus<digit>> {};

struct add_op : sor<one<'+'>, one<'-'>> {};
struct mul_op : sor<one<'*'>, one<'/'>> {};

struct factor : sor<integer, identifier> {};
struct term : seq<factor, star<seq<ws, mul_op, ws, factor>>> {};

struct expression : seq<term, star<seq<ws, add_op, ws, term>>> {};
struct assignment : seq<identifier, ws, one<'='>, ws, expression> {};

struct statement : seq<sor<assignment, expression>> {};
struct end_of_statement : plus<one<'\n', '\r', ';'>> {};

struct start : seq<statement, star<seq<end_of_statement, statement >>> {};

//struct start : seq<sor<assignment, expression>, plus<one<'\n', '\r'>>> {};
} // namespace grammar
