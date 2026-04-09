# ============================================================
# complete_mcp_server.py
# Shows ALL THREE MCP capabilities:
#   1. Tools     — functions Claude can CALL
#   2. Resources — data Claude can READ
#   3. Prompts   — templates Claude can USE
#
# Run in terminal: python complete_mcp_server.py
# ============================================================

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("complete_server")


# ============================================================
# 1. TOOLS — Functions Claude can CALL
#    Claude sends arguments → function runs → result returned
#    Decorated with @mcp.tool()
# ============================================================

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return the result."""
    return a * b

@mcp.tool()
def get_student_grade(student_name: str) -> str:
    """Get the grade of a student by their name."""
    grades = {
        "ali":   "A+",
        "sara":  "B+",
        "ahmed": "A",
        "zara":  "B",
    }
    grade = grades.get(student_name.lower())
    if grade:
        return f"{student_name.title()} has grade: {grade}"
    return f"Student '{student_name}' not found."


# ============================================================
# 2. RESOURCES — Data Claude can READ
#    Like files or database records the server exposes
#    Claude reads them as context (not by calling a function)
#    Decorated with @mcp.resource("resource://name")
# ============================================================

@mcp.resource("resource://school-info")
def get_school_info() -> str:
    """Basic information about the school."""
    return """
    School Name : Punjab AI Academy
    Principal   : Dr. Ahmed Khan
    Students    : 500
    Courses     : Python, AI, Data Science, MCP
    Location    : Lahore, Pakistan
    """

@mcp.resource("resource://class-schedule")
def get_class_schedule() -> str:
    """The weekly class schedule."""
    return """
    Monday    : Python Basics       9:00 AM
    Tuesday   : AI Fundamentals    10:00 AM
    Wednesday : LangChain Tools    11:00 AM
    Thursday  : MCP Architecture    9:00 AM
    Friday    : Project Work       10:00 AM
    """

@mcp.resource("resource://student/{name}")
def get_student_resource(name: str) -> str:
    """Get detailed info about a specific student."""
    students = {
        "ali":   {"grade": "A+", "age": 20, "course": "AI"},
        "sara":  {"grade": "B+", "age": 22, "course": "Python"},
        "ahmed": {"grade": "A",  "age": 21, "course": "Data Science"},
    }
    info = students.get(name.lower())
    if info:
        return f"Name: {name.title()}, Grade: {info['grade']}, Age: {info['age']}, Course: {info['course']}"
    return f"Student '{name}' not found."


# ============================================================
# 3. PROMPTS — Pre-written templates Claude can USE
#    Ready-made prompt structures for common tasks
#    Claude fills in the variables and uses them
#    Decorated with @mcp.prompt()
# ============================================================

@mcp.prompt()
def explain_concept(concept: str) -> str:
    """A prompt template for explaining any AI concept simply."""
    return f"""
    You are a helpful AI teacher.
    Explain the concept of '{concept}' in very simple terms.
    Use a real-world analogy.
    Keep it under 5 sentences.
    Make it easy for a beginner to understand.
    """

@mcp.prompt()
def grade_report(student_name: str, grade: str) -> str:
    """A prompt template for writing a student grade report."""
    return f"""
    Write a short, encouraging grade report for student '{student_name}'
    who received a grade of '{grade}'.
    Include:
    - One sentence congratulating or encouraging them
    - One specific suggestion for improvement
    - A positive closing statement
    Keep the tone warm and professional.
    """

@mcp.prompt()
def debug_code(code: str, error: str) -> str:
    """A prompt template for debugging Python code."""
    return f"""
    You are an expert Python debugger.
    Here is the code:
    ```python
    {code}
    ```
    Here is the error:
    ```
    {error}
    ```
    Please:
    1. Explain what caused the error in simple terms
    2. Show the fixed code
    3. Explain what you changed and why
    """


# ============================================================
# START THE SERVER
# ============================================================
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
