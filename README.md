This small repo is a sample of our integration with a LLM Vendor. Your task is to implement integration with Amazon Bedrock hosted Claude 3 Sonnet by deriving bedrock_claude3_call.Model from the class llm_call.LLM, and implementing it's functionality.

To test, copy file tests/test_together_llama.py to tests/test_bedrock_claude3.py, replace the code with calls to claude3, and record the results. (hint: make sure your zone is set to us-east-2 and append "us." to the start of the bedrock model string)

The deliverables are:

Demonstrated functionality
What RPM can we hit in the parallel asynchronous execution, and what is the optimal parallelism?
What is the error rate?
Do we need to change the system prompt or the API call parameters to achieve higher success rate, without changeing the user questions?
