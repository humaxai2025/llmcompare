function llm-compare {
    param([Parameter(ValueFromRemainingArguments)]$Args)
    Set-Location "C:\Users\Sriram\llm-comparator"
    & "C:\Users\Sriram\llm-comparator\venv\Scripts\python.exe" "C:\Users\Sriram\llm-comparator\llmcompare.py" @Args
}

Write-Host "LLM Comparator activated. Try: llm-compare --help" -ForegroundColor Green
