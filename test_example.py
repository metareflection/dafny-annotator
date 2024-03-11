program = '''
method intersperse(numbers: seq<int>, delimiter: int) returns (interspersed: seq<int>)
    ensures |interspersed| == if |numbers| > 0 then 2 * |numbers| - 1 else 0
    ensures forall i :: 0 <= i < |interspersed| ==> i % 2 == 0 ==>
                interspersed[i] == numbers[i / 2]
    ensures forall i :: 0 <= i < |interspersed| ==> i % 2 == 1 ==>
                interspersed[i] == delimiter
{
    interspersed := [];
    for i := 0 to |numbers|
    {
        if i > 0 {
            interspersed := interspersed + [delimiter];
        }
        interspersed := interspersed + [numbers[i]];
    }
}'''.strip()

verification_prompt = """Given the following Dafny program:
Add an assertion or an invariant in order to verify the program.

Program 1:
method maxArray(a: array<int>) returns (m: int)
  requires a.Length >= 1
  ensures forall k :: 0 <= k < a.Length ==> m >= a[k]
  ensures exists k :: 0 <= k < a.Length && m == a[k]
{
  m := a[0];
  var index := 1;
  while (index < a.Length)
     decreases a.Length - index
  {
    m := if m>a[index] then  m else a[index];
    index := index + 1;
  }
}

Action 1: // On line 9, add invariant 0 <= index <= a.Length;

Program 2:
"""+program+"""

Action 2:"""
