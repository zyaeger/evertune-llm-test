This small repo is a sample of our integration with a LLM Vendor. Your task is to implement integration with Google Vertex hosted gemini 2.5 flash by deriving gemini_call.Model from the class llm_call.LLM, and implementing it's functionality.

To test, copy file tests/test_together_llama.py to tests/test_gemini_2_5_flash.py, replace the code with calls to gemini, and record the results.(you will need yo download and install the gcp cli and configure it to the project evertune-tests and location us-central1

The deliverables are:

Demonstrated functionality
What RPM can we hit in the parallel asynchronous execution, and what is the optimal parallelism?
What is the error rate?
Do we need to change the system prompt or the API call parameters to achieve higher success rate, without changeing the user questions?
