This small repo is a sample of our integration with a LLM Vendor. Your task is to implement integration with Google Vertex hosted gemini 2.5 flash by deriving gemini_call.Model from the class llm_call.LLM, and implementing it's functionality.

To test, copy file tests/test_together_llama.py to tests/test_gemini_2_5_flash.py, replace the code with calls to gemini, and record the results.(you will need yo download and install the gcp cli and configure it to the project evertune-tests and location us-central1

The deliverables are:

Demonstrated functionality
What RPM can we hit in the parallel asynchronous execution, and what is the optimal parallelism?
What is the error rate?
Do we need to change the system prompt or the API call parameters to achieve higher success rate, without changeing the user questions?

### Findings

> What RPM can we hit in the parallel asynchronous execution, and what is the optimal parallelism?

I was able to achieve upwards of 40k RPM, with a range of 10k-20k parallelism, with closer to 15k being the "sweet spot".

> What is the error rate?

In terms of "bad data" being returned or certain errors being handled, the error rate is virtually 0. In terms of errors happening that were not handled, I got throttling errors in about 5% of tests, growing as the parallelism I tested grew past 20k.

> Do we need to change the system prompt or the API call parameters to achieve higher success rate, without changeing the user questions?

For the choices system prompt, I found success keeping as is. No return data was labelled as "Bad" during testing. 

As for the ranked list prompt, I found greater success by specifying it should be ranking brands, rather than products. I.e., Gemini was consistently confusing Range Rover as "different" than Land Rover, having different brands like:
Range Rover, Land Rover, Range Rover (Land Rover), and Land Rover (Range Rover). This is an example, but there were others (like BMW's X-line). This was done without changing the user questions.

By specifying to the model that we were ranking brands, not products, the output was cleaned up significantly, with _one_ exception: Range Rover/Land Rover. Although the parenthesized brands had mostly disappeared, both were still present in the responses, so much so that Land Rover was ranked 1st, and Range Rover 2nd or 3rd, depending on the run.

#### Before

| Brand                               |    #1 |    #2 |    #3 |    #4 |    #5 |
|-------------------------------------|-------|-------|-------|-------|-------|
| Land Rover                          | 27785 |   165 |  1787 |  7380 |   134 |
| Range Rover                         | 18773 |    63 |  1444 |  2051 |   115 |
| Mercedes-Benz                       | 12337 | 42022 |  3770 |   435 |   252 |
| Rolls-Royce                         |   932 |    12 |     3 |     4 |   158 |
| Cadillac                            |    99 |   607 |  8682 |  2015 |  7779 |
| Land Rover (Range Rover)            |    51 |     0 |     7 |     2 |     1 |
| BMW                                 |    11 | 12334 | 32181 |  7334 |  4334 |
| Porsche                             |     6 |  2776 |  1326 | 16995 | 23088 |
| Rolls-Royce Cullinan                |     3 |     5 |    39 |    36 |    78 |
| Land Rover/Range Rover              |     2 |     0 |     1 |     0 |     0 |
| Bentley                             |     1 |   935 |    20 |    44 |   640 |
| Mercedes-Benz G-Class               |     0 |   762 |   257 |     6 |    14 |
| Cadillac Escalade                   |     0 |   281 |   630 |    49 |    31 |
| Mercedes-Benz (G-Class/GLS)         |     0 |    10 |     0 |     0 |     0 |
| Mercedes-Benz G-Wagen               |     0 |     8 |     4 |     0 |     0 |
| Mercedes-Benz GLS                   |     0 |     8 |     0 |     0 |     0 |
| Mercedes-Benz (G-Wagen/GLS)         |     0 |     4 |     0 |     0 |     0 |
| Mercedes-Benz (G-Class)             |     0 |     4 |     0 |     0 |     0 |
| Bentley Bentayga                    |     0 |     3 |    40 |    45 |    33 |
| Lexus                               |     0 |     1 |     9 |  1557 | 14540 |
| Audi                                |     0 |     0 |  9238 | 20811 |  7786 |
| Lamborghini                         |     0 |     0 |   335 |    47 |    14 |
| Aston Martin                        |     0 |     0 |    83 |   218 |    31 |
| BMW X7                              |     0 |     0 |    76 |   115 |   461 |
| Porsche Cayenne                     |     0 |     0 |    45 |   812 |   180 |
| Cadillac (Escalade)                 |     0 |     0 |    15 |     2 |     5 |
| Mercedes-Benz G-Class (G-Wagen)     |     0 |     0 |     4 |     0 |     0 |
| Porsche (Cayenne)                   |     0 |     0 |     2 |     8 |     6 |
| BMW (X Series)                      |     0 |     0 |     1 |     2 |     0 |
| Land Rover Range Rover              |     0 |     0 |     1 |     1 |     0 |
| Lexus LX                            |     0 |     0 |     0 |     8 |   220 |
| Mercedes-Benz (Maybach GLS)         |     0 |     0 |     0 |     6 |     1 |
| Audi Q8                             |     0 |     0 |     0 |     4 |    17 |
| BMW (X5/X7)                         |     0 |     0 |     0 |     4 |     1 |
| BMW (X7)                            |     0 |     0 |     0 |     3 |     0 |
| Lamborghini Urus                    |     0 |     0 |     0 |     1 |     6 |
| Ferrari                             |     0 |     0 |     0 |     1 |     5 |
| BMW X Series                        |     0 |     0 |     0 |     1 |     3 |
| Mercedes-Benz (Maybach/G-Class)     |     0 |     0 |     0 |     1 |     1 |
| Range Rover SV                      |     0 |     0 |     0 |     1 |     0 |
| BMW (X7/X5)                         |     0 |     0 |     0 |     1 |     0 |
| Mercedes-Benz AMG                   |     0 |     0 |     0 |     0 |    17 |
| Aston Martin DBX                    |     0 |     0 |     0 |     0 |    15 |
| BMW X5                              |     0 |     0 |     0 |     0 |    14 |
| Lexus (LX)                          |     0 |     0 |     0 |     0 |     6 |
| Mercedes-Maybach                    |     0 |     0 |     0 |     0 |     3 |
| BMW X5/X7                           |     0 |     0 |     0 |     0 |     3 |
| Mercedes-Benz (G-Class/Maybach GLS) |     0 |     0 |     0 |     0 |     2 |
| BMW X-Series                        |     0 |     0 |     0 |     0 |     2 |
| Bentley (Bentayga)                  |     0 |     0 |     0 |     0 |     1 |
| Mercedes-Benz Maybach               |     0 |     0 |     0 |     0 |     1 |
| Land Rover (Range Rover SV)         |     0 |     0 |     0 |     0 |     1 |
| Mercedes-Benz AMG G-Class           |     0 |     0 |     0 |     0 |     1 |

#### After

| Brand                    |    #1 |    #2 |    #3 |    #4 |    #5 |
|--------------------------|-------|-------|-------|-------|-------|
| Land Rover               | 36296 |  1170 |  4691 | 11323 |   352 |
| Mercedes-Benz            | 20724 | 37738 |  1450 |    73 |    10 |
| Range Rover              |  2714 |   121 |  1248 |  1940 |   125 |
| BMW                      |   135 | 19718 | 34512 |  4299 |   916 |
| Rolls-Royce              |   116 |     0 |     0 |     3 |    14 |
| Cadillac                 |    11 |     7 |  1691 |  1158 |  4765 |
| Land Rover (Range Rover) |     2 |     0 |     0 |     0 |     0 |
| Porsche                  |     1 |  1130 |  2690 | 17059 | 11281 |
| Lexus                    |     1 |     0 |    23 |  4727 | 35095 |
| Bentley                  |     0 |   116 |     3 |    17 |   161 |
| Audi                     |     0 |     0 | 13663 | 19394 |  7262 |
| Lamborghini              |     0 |     0 |    29 |     1 |    16 |
| Aston Martin             |     0 |     0 |     0 |     6 |     3 |

(As you can see, the "Land Rover (Range Rover)" problem showed its face.)
The different prompts used are in `constants.py`, the original is commented out.

I did not mention changing API parameters....

## Note

I made some design changes to the repo. Everything still functions as intended, but cleaned things up a bit, like moving the shared resources to `constants.py`.
Also added pylint and black formatting for my own readability.