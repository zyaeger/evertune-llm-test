This small repo is a sample of our integration with a LLM Vendor. Your task is to implement integration with {vendor} hosted {model} by deriving {vendor}_{model_shortname}_call.Model from the class llm_call.LLM, and implementing it's functionality.

To test, copy file tests/test_together_llama.py to tests/test_{vendor}_{model_shortname}.py, replace the code with calls to Claude, and record the results.

The deliverables are:

Demonstrated functionality
What RPM can we hit in parallel asynchronous execution and what is the optimal parallelism?
What is the error rate?
Do we need to change the system prompt or the API call parameters to achieve higher success rate? We cannot change the questions.
