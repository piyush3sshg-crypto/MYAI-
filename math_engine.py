import re
import math

TOKEN_PATTERN = re.compile(r"\s*(?:(\d+\.\d+|\d+)|([A-Za-z_][A-Za-z0-9_]*)|(\*\*|[+\-*/%^()=]))")


class MathToken:
    __slots__ = ("kind", "value")

    def __init__(self, kind, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"Token({self.kind}, {self.value})"


def tokenize_expression(expression):
    tokens = []
    position = 0
    while position < len(expression):
        match = TOKEN_PATTERN.match(expression, position)
        if not match or match.end() == position:
            if expression[position].isspace():
                position += 1
                continue
            raise ValueError(f"unexpected character at position {position}: {expression[position]!r}")
        number, identifier, operator = match.groups()
        if number is not None:
            tokens.append(MathToken("NUMBER", float(number)))
        elif identifier is not None:
            tokens.append(MathToken("IDENT", identifier))
        elif operator is not None:
            tokens.append(MathToken("OP", operator))
        position = match.end()
    return tokens


class ASTNode:
    pass


class NumberNode(ASTNode):
    def __init__(self, value):
        self.value = value

    def evaluate(self, environment):
        return self.value


class VariableNode(ASTNode):
    def __init__(self, name):
        self.name = name

    def evaluate(self, environment):
        if self.name not in environment:
            raise KeyError(f"undefined variable {self.name}")
        return environment[self.name]


class BinaryOpNode(ASTNode):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, environment):
        left_val = self.left.evaluate(environment)
        right_val = self.right.evaluate(environment)
        if self.op == "+":
            return left_val + right_val
        if self.op == "-":
            return left_val - right_val
        if self.op == "*":
            return left_val * right_val
        if self.op == "/":
            return left_val / right_val
        if self.op == "%":
            return left_val % right_val
        if self.op in ("^", "**"):
            return left_val ** right_val
        raise ValueError(f"unsupported operator {self.op}")


class UnaryOpNode(ASTNode):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def evaluate(self, environment):
        value = self.operand.evaluate(environment)
        if self.op == "-":
            return -value
        return value


class FunctionCallNode(ASTNode):
    def __init__(self, name, argument):
        self.name = name
        self.argument = argument

    def evaluate(self, environment):
        arg_value = self.argument.evaluate(environment)
        functions = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "abs": abs,
            "exp": math.exp,
            "floor": math.floor,
            "ceil": math.ceil,
        }
        if self.name not in functions:
            raise ValueError(f"unknown function {self.name}")
        return functions[self.name](arg_value)


class ExpressionParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0

    def _current(self):
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def _advance(self):
        token = self._current()
        self.position += 1
        return token

    def _expect_op(self, value):
        token = self._current()
        if token is None or token.kind != "OP" or token.value != value:
            raise ValueError(f"expected {value!r} got {token!r}")
        return self._advance()

    def parse(self):
        node = self.parse_expression()
        if self._current() is not None:
            raise ValueError(f"unexpected trailing tokens starting at {self._current()!r}")
        return node

    def parse_expression(self):
        return self.parse_additive()

    def parse_additive(self):
        node = self.parse_multiplicative()
        while True:
            token = self._current()
            if token is not None and token.kind == "OP" and token.value in ("+", "-"):
                self._advance()
                right = self.parse_multiplicative()
                node = BinaryOpNode(token.value, node, right)
            else:
                break
        return node

    def parse_multiplicative(self):
        node = self.parse_exponent()
        while True:
            token = self._current()
            if token is not None and token.kind == "OP" and token.value in ("*", "/", "%"):
                self._advance()
                right = self.parse_exponent()
                node = BinaryOpNode(token.value, node, right)
            else:
                break
        return node

    def parse_exponent(self):
        node = self.parse_unary()
        token = self._current()
        if token is not None and token.kind == "OP" and token.value in ("^", "**"):
            self._advance()
            right = self.parse_exponent()
            node = BinaryOpNode(token.value, node, right)
        return node

    def parse_unary(self):
        token = self._current()
        if token is not None and token.kind == "OP" and token.value in ("+", "-"):
            self._advance()
            operand = self.parse_unary()
            return UnaryOpNode(token.value, operand)
        return self.parse_primary()

    def parse_primary(self):
        token = self._current()
        if token is None:
            raise ValueError("unexpected end of expression")
        if token.kind == "NUMBER":
            self._advance()
            return NumberNode(token.value)
        if token.kind == "IDENT":
            self._advance()
            next_token = self._current()
            if next_token is not None and next_token.kind == "OP" and next_token.value == "(":
                self._advance()
                argument = self.parse_expression()
                self._expect_op(")")
                return FunctionCallNode(token.value, argument)
            return VariableNode(token.value)
        if token.kind == "OP" and token.value == "(":
            self._advance()
            node = self.parse_expression()
            self._expect_op(")")
            return node
        raise ValueError(f"unexpected token {token!r}")


def evaluate_expression(expression, environment=None):
    environment = environment or {}
    tokens = tokenize_expression(expression)
    parser = ExpressionParser(tokens)
    ast = parser.parse()
    return ast.evaluate(environment)


def parse_linear_equation(equation_text, variable="x"):
    if "=" not in equation_text:
        raise ValueError("equation must contain '='")
    left_text, right_text = equation_text.split("=", 1)
    left_coef, left_const = _linear_coefficients(left_text, variable)
    right_coef, right_const = _linear_coefficients(right_text, variable)
    coef = left_coef - right_coef
    const = right_const - left_const
    if coef == 0:
        if const == 0:
            return "infinite_solutions"
        return "no_solution"
    return const / coef


def _linear_coefficients(text, variable):
    tokens = tokenize_expression(text)
    coef = 0.0
    const = 0.0
    sign = 1.0
    i = 0
    pending_number = None
    while i < len(tokens):
        token = tokens[i]
        if token.kind == "OP" and token.value == "+":
            sign = 1.0
            pending_number = None
            i += 1
            continue
        if token.kind == "OP" and token.value == "-":
            sign = -1.0
            pending_number = None
            i += 1
            continue
        if token.kind == "NUMBER":
            pending_number = token.value
            if i + 1 < len(tokens) and tokens[i + 1].kind == "OP" and tokens[i + 1].value == "*":
                i += 2
                continue
            if i + 1 < len(tokens) and tokens[i + 1].kind == "IDENT" and tokens[i + 1].value == variable:
                coef += sign * pending_number
                pending_number = None
                i += 2
                continue
            const += sign * pending_number
            pending_number = None
            i += 1
            continue
        if token.kind == "IDENT" and token.value == variable:
            multiplier = pending_number if pending_number is not None else 1.0
            coef += sign * multiplier
            pending_number = None
            i += 1
            continue
        i += 1
    return coef, const


def solve_quadratic(a, b, c):
    if a == 0:
        if b == 0:
            return []
        return [-c / b]
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        real_part = -b / (2 * a)
        imaginary_part = math.sqrt(-discriminant) / (2 * a)
        return [complex(real_part, imaginary_part), complex(real_part, -imaginary_part)]
    if discriminant == 0:
        return [-b / (2 * a)]
    sqrt_discriminant = math.sqrt(discriminant)
    return [
        (-b + sqrt_discriminant) / (2 * a),
        (-b - sqrt_discriminant) / (2 * a),
    ]


class ArithmeticReasoner:
    def __init__(self):
        self.variable_bindings = {}

    def define_variable(self, name, value):
        self.variable_bindings[name] = value

    def evaluate(self, expression):
        return evaluate_expression(expression, self.variable_bindings)

    def solve_for(self, equation_text, variable="x"):
        return parse_linear_equation(equation_text, variable)

    def solve_quadratic_equation(self, a, b, c):
        return solve_quadratic(a, b, c)

