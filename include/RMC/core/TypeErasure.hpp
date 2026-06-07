#pragma once
// Generator macros for the project's owning type-erasure idiom
// (concept + Concept-ABC + Model<T> + passkey, with value semantics via
// clone()). Each wrapper Foo is declared by listing only its methods; the
// derived names CFoo / detail::FooToken / FooConcept / FooModel follow the
// existing convention, so the generator needs nothing but the wrapper name.
//
// A method is described by a Boost.PP tuple. Normal row (9 fields):
//   (NODISCARD, RET, NAME, PARAMS, ARGC, ARGS, CV, NOEXCEPT, TOKEN)
//     NODISCARD : 0/1   -> emits [[nodiscard]] when 1
//     RET       : return type
//     NAME      : method name
//     PARAMS    : parenthesized typed parameter list, emitted verbatim, e.g.
//                 (const coords_t &c, std::span<const std::size_t> m) or ()
//     ARGC      : number of forward args (explicit, avoids empty-tuple
//     pitfalls) ARGS      : parenthesized bare arg names, e.g. (c, m) or () CV
//     : const  or empty NOEXCEPT  : noexcept or empty TOKEN     : WITH_TOKEN
//     (forward data_.f(make_token(), args...)) or
//                 NO_TOKEN  (forward data_.f(args...))
//
// Optional row (10 fields) — for methods gated on a refining concept that fall
// back to a default when the wrapped type does not provide them:
//   (NODISCARD, RET, NAME, PARAMS, ARGC, ARGS, CV, NOEXCEPT, REFINES, DEFAULT)
//     REFINES   : refining concept (called as REFINES<T>)
//     DEFAULT   : expression returned when REFINES<T> is not satisfied
//   Optional methods forward to data_ WITHOUT the passkey token (mirrors the
//   existing CMoveGeneratorWith* concepts, which require token-free calls).
//
// Lists are Boost.PP sequences of these tuples. Build a wrapper with
// RMC_DEFINE_ERASED_TYPE(Foo, METHODS) or, when optional rows are present,
// RMC_DEFINE_ERASED_TYPE_EXT(Foo, METHODS, OPT_METHODS).

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/enum.hpp>
#include <memory>
#include <utility>

// --- field accessors --------------------------------------------------------
#define RMC_TE_ND_F(M) BOOST_PP_TUPLE_ELEM(0, M)
#define RMC_TE_RET(M) BOOST_PP_TUPLE_ELEM(1, M)
#define RMC_TE_NAME(M) BOOST_PP_TUPLE_ELEM(2, M)
#define RMC_TE_PARAMS(M) BOOST_PP_TUPLE_ELEM(3, M)
#define RMC_TE_ARGC(M) BOOST_PP_TUPLE_ELEM(4, M)
#define RMC_TE_ARGS_F(M) BOOST_PP_TUPLE_ELEM(5, M)
#define RMC_TE_CV(M) BOOST_PP_TUPLE_ELEM(6, M)
#define RMC_TE_NX(M) BOOST_PP_TUPLE_ELEM(7, M)
#define RMC_TE_TOK(M) BOOST_PP_TUPLE_ELEM(8, M)
#define RMC_TE_REFINES(M) BOOST_PP_TUPLE_ELEM(8, M)
#define RMC_TE_DEFAULT(M) BOOST_PP_TUPLE_ELEM(9, M)

// --- small helpers ----------------------------------------------------------
#define RMC_TE_EAT(...)
#define RMC_TE_ND(ND) BOOST_PP_IF(ND, [[nodiscard]], )
// Token-mode -> boolean for BOOST_PP_IF.
#define RMC_TE_TOKEN_WITH_TOKEN 1
#define RMC_TE_TOKEN_NO_TOKEN 0
#define RMC_TE_TOKBOOL(M) BOOST_PP_CAT(RMC_TE_TOKEN_, RMC_TE_TOK(M))

// Enumerate forward args ("c, m"); empty when ARGC == 0.
#define RMC_TE_ENUM_ARGS(ARGC, ARGS)                                           \
  BOOST_PP_IF(ARGC, BOOST_PP_TUPLE_ENUM, RMC_TE_EAT)(ARGS)

// Forwarding argument list for a Model override, with or without the passkey.
#define RMC_TE_FWD_WITH(M)                                                     \
  make_token() BOOST_PP_COMMA_IF(RMC_TE_ARGC(M))                               \
      RMC_TE_ENUM_ARGS(RMC_TE_ARGC(M), RMC_TE_ARGS_F(M))
#define RMC_TE_FWD_NO(M) RMC_TE_ENUM_ARGS(RMC_TE_ARGC(M), RMC_TE_ARGS_F(M))
#define RMC_TE_FWD(M)                                                          \
  BOOST_PP_IF(RMC_TE_TOKBOOL(M), RMC_TE_FWD_WITH, RMC_TE_FWD_NO)(M)

// --- per-method generators (driven by BOOST_PP_SEQ_FOR_EACH) ----------------
// Public wrapper forwarder (delegates to the type-erased self_). Reused for
// optional rows too — it only reads fields 0..7.
#define RMC_TE_WRAPPER_METHOD(r, data, M)                                      \
  RMC_TE_ND(RMC_TE_ND_F(M))                                                    \
  constexpr RMC_TE_RET(M) RMC_TE_NAME(M) RMC_TE_PARAMS(M) RMC_TE_CV(M)         \
      RMC_TE_NX(M) {                                                           \
    return self_->RMC_TE_NAME(M)(                                              \
        RMC_TE_ENUM_ARGS(RMC_TE_ARGC(M), RMC_TE_ARGS_F(M)));                   \
  }

// Pure-virtual entry in the Concept ABC.
#define RMC_TE_CONCEPT_METHOD(r, data, M)                                      \
  RMC_TE_ND(RMC_TE_ND_F(M))                                                    \
  virtual RMC_TE_RET(M) RMC_TE_NAME(M) RMC_TE_PARAMS(M) RMC_TE_CV(M)           \
      RMC_TE_NX(M) = 0;

// Concrete override in Model<T>, forwarding into the wrapped value.
#define RMC_TE_MODEL_METHOD(r, data, M)                                        \
  constexpr RMC_TE_RET(M) RMC_TE_NAME(M) RMC_TE_PARAMS(M) RMC_TE_CV(M)         \
      RMC_TE_NX(M) override {                                                  \
    return data_.RMC_TE_NAME(M)(RMC_TE_FWD(M));                                \
  }

// Optional-method Concept entry: non-pure virtual with a default body.
#define RMC_TE_OPT_CONCEPT_METHOD(r, data, M)                                  \
  RMC_TE_ND(RMC_TE_ND_F(M))                                                    \
  virtual RMC_TE_RET(M) RMC_TE_NAME(M) RMC_TE_PARAMS(M) RMC_TE_CV(M)           \
      RMC_TE_NX(M) {                                                           \
    return RMC_TE_DEFAULT(M);                                                  \
  }

// Optional-method Model override: forward when REFINES<T> holds, else default.
#define RMC_TE_OPT_MODEL_METHOD(r, data, M)                                    \
  constexpr RMC_TE_RET(M) RMC_TE_NAME(M) RMC_TE_PARAMS(M) RMC_TE_CV(M)         \
      RMC_TE_NX(M) override {                                                  \
    if constexpr (RMC_TE_REFINES(M) < T >) {                                   \
      return data_.RMC_TE_NAME(M)(                                             \
          RMC_TE_ENUM_ARGS(RMC_TE_ARGC(M), RMC_TE_ARGS_F(M)));                 \
    } else {                                                                   \
      return RMC_TE_DEFAULT(M);                                                \
    }                                                                          \
  }

// --- shared boilerplate -----------------------------------------------------
// The converting ctor + the four copy/move special members, identical across
// every wrapper.
#define RMC_TE_SPECIAL_MEMBERS(Name)                                           \
  using Token = detail::BOOST_PP_CAT(Name, Token);                             \
  template <BOOST_PP_CAT(C, Name) T>                                           \
  constexpr Name(T x)                                                          \
      : self_(std::make_unique < BOOST_PP_CAT(Name, Model) < T >>              \
              (std::move(x))) {}                                               \
  constexpr Name(const Name &s) : self_{s.self_->clone()} {}                   \
  constexpr Name(Name &&s) noexcept : self_{std::move(s.self_)} {}             \
  constexpr Name &operator=(const Name &s) {                                   \
    self_ = s.self_->clone();                                                  \
    return *this;                                                              \
  }                                                                            \
  constexpr Name &operator=(Name &&s) noexcept {                               \
    self_ = std::move(s.self_);                                                \
    return *this;                                                              \
  }

// --- assembly ---------------------------------------------------------------
#define RMC_DEFINE_ERASED_TYPE(Name, METHODS)                                  \
  RMC_TE_BEGIN(Name)                                                           \
  BOOST_PP_SEQ_FOR_EACH(RMC_TE_WRAPPER_METHOD, ~, METHODS)                     \
  RMC_TE_PRIVATE(Name)                                                         \
  struct BOOST_PP_CAT(Name, Concept) {                                         \
    virtual ~BOOST_PP_CAT(Name, Concept)() = default;                          \
    BOOST_PP_SEQ_FOR_EACH(RMC_TE_CONCEPT_METHOD, ~, METHODS)                   \
    virtual std::unique_ptr<BOOST_PP_CAT(Name, Concept)> clone() const = 0;    \
  };                                                                           \
  template <BOOST_PP_CAT(C, Name) T>                                           \
  struct BOOST_PP_CAT(Name, Model) final : BOOST_PP_CAT(Name, Concept) {       \
    constexpr explicit BOOST_PP_CAT(Name, Model)(T x) : data_(std::move(x)) {} \
    BOOST_PP_SEQ_FOR_EACH(RMC_TE_MODEL_METHOD, ~, METHODS)                     \
    RMC_TE_CLONE(Name)                                                         \
    T data_;                                                                   \
  };                                                                           \
  RMC_TE_END(Name)

#define RMC_DEFINE_ERASED_TYPE_EXT(Name, METHODS, OPT)                         \
  RMC_TE_BEGIN(Name)                                                           \
  BOOST_PP_SEQ_FOR_EACH(RMC_TE_WRAPPER_METHOD, ~, METHODS)                     \
  BOOST_PP_SEQ_FOR_EACH(RMC_TE_WRAPPER_METHOD, ~, OPT)                         \
  RMC_TE_PRIVATE(Name)                                                         \
  struct BOOST_PP_CAT(Name, Concept) {                                         \
    virtual ~BOOST_PP_CAT(Name, Concept)() = default;                          \
    BOOST_PP_SEQ_FOR_EACH(RMC_TE_CONCEPT_METHOD, ~, METHODS)                   \
    BOOST_PP_SEQ_FOR_EACH(RMC_TE_OPT_CONCEPT_METHOD, ~, OPT)                   \
    virtual std::unique_ptr<BOOST_PP_CAT(Name, Concept)> clone() const = 0;    \
  };                                                                           \
  template <BOOST_PP_CAT(C, Name) T>                                           \
  struct BOOST_PP_CAT(Name, Model) final : BOOST_PP_CAT(Name, Concept) {       \
    constexpr explicit BOOST_PP_CAT(Name, Model)(T x) : data_(std::move(x)) {} \
    BOOST_PP_SEQ_FOR_EACH(RMC_TE_MODEL_METHOD, ~, METHODS)                     \
    BOOST_PP_SEQ_FOR_EACH(RMC_TE_OPT_MODEL_METHOD, ~, OPT)                     \
    RMC_TE_CLONE(Name)                                                         \
    T data_;                                                                   \
  };                                                                           \
  RMC_TE_END(Name)

// Structural fragments shared by both assemblers.
#define RMC_TE_BEGIN(Name)                                                     \
  class Name {                                                                 \
  public:                                                                      \
    RMC_TE_SPECIAL_MEMBERS(Name)
#define RMC_TE_PRIVATE(Name)                                                   \
private:                                                                       \
  static Token make_token() noexcept { return {}; }
#define RMC_TE_CLONE(Name)                                                     \
  constexpr std::unique_ptr<BOOST_PP_CAT(Name, Concept)> clone()               \
      const override {                                                         \
    return std::make_unique<BOOST_PP_CAT(Name, Model)>(data_);                 \
  }
#define RMC_TE_END(Name)                                                       \
  std::unique_ptr<BOOST_PP_CAT(Name, Concept)> self_;                          \
  }                                                                            \
  ;
