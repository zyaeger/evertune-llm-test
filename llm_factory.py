import gemini_call
import gpt_call
import perplexity_call
import together_llm_call

class LLMmodel:
    @staticmethod
    def get_model(model_name: str):
        match model_name:
            case _ if model_name in together_llm_call.Model.known_models():
                model = together_llm_call.Model()
            case _ if model_name in gpt_call.Model.known_models():
                model = gpt_call.Model(model_name)
            case _ if model_name in gemini_call.Model.known_models():
                model = gemini_call.Model(model_name)
            case _ if model_name in perplexity_call.PerplexityModel.known_models():
                model = perplexity_call.PerplexityModel(model_name)
            case _: raise ValueError(f"Unknown model {model_name}")
        parallelism = model.parallelism()

        return model, parallelism

    @staticmethod
    def known_models() -> set[str]:
        return set([m for m in
                    gpt_call.Model.known_models() |
                    gemini_call.Model.known_models() |
                    together_llm_call.Model.known_models() |
                    perplexity_call.PerplexityModel.known_models()
                    if not any(m.startswith(bad_model) for bad_model in [
                        'babbage', 'dall-e', 'davinci', 'mistral', 'mixtral', 'text-embedding', 'tts-', 'whisper', 'codellama'
                ])])

    @staticmethod
    def latest_models(models: list[str]) -> dict[str, tuple[str, str]]:
        result = {}
        gpt = gpt_call.Model.order_models(models)
        if len(gpt) > 0:
            result['Open AI'] = (gpt[0], gpt_call.Model.display_name(gpt[0]))
        gemini = gemini_call.Model.order_models(models)
        if len(gemini) > 0:
            result['Google'] = (gemini[0], gemini_call.Model.display_name(gemini[0]))
        llama = together_llm_call.Model.order_models(models)
        if len(llama) > 0:
            result['Meta'] = (llama[0], together_llm_call.Model.display_name(llama[0]))
        # llama = perplexity_call.PerplexityModel.order_llama_models(models)
        # if len(llama) > 0:
        #     result['Meta'] = (llama[0], perplexity_call.PerplexityModel.display_name(llama[0]))
        perplexity = perplexity_call.PerplexityModel.order_perplexity_models(models)
        if len(perplexity) > 0:
            result['Perplexity'] = (perplexity[0], perplexity_call.PerplexityModel.display_name(perplexity[0]))
        return result

    @staticmethod
    def display_name(model_name: str) -> tuple[str, str]:
        match model_name:
            case _ if model_name in gpt_call.Model.known_models():
                return 'Open AI', gpt_call.Model.display_name(model_name)
            case _ if model_name in gemini_call.Model.known_models():
                return 'Google', gemini_call.Model.display_name(model_name)
            case _ if model_name in together_llm_call.Model.known_models():
                return 'Meta', together_llm_call.Model.display_name(model_name)
            case _ if model_name in perplexity_call.PerplexityModel.known_models():
                vendor = perplexity_call.PerplexityModel.model_vendor(model_name)
                return vendor, perplexity_call.PerplexityModel.display_name(model_name)
            case _: raise ValueError(f"Unknown model {model_name}")

    @staticmethod
    def sort_by_latest_models(models: list[str]) -> list[tuple[str, tuple[str, str]]]:
        gpt = gpt_call.Model.order_models(models)
        gemini = gemini_call.Model.order_models(models)
        llama = together_llm_call.Model.order_models(models)
        perplexity = perplexity_call.PerplexityModel.order_perplexity_models(models)
        return ([('Open AI', (m, gpt_call.Model.display_name(m))) for m in gpt[:1]] +
                [('Google', (m, gemini_call.Model.display_name(m))) for m in gemini[:1]] +
                [('Meta', (m, together_llm_call.Model.display_name(m))) for m in llama[:1]] +
                [('Perplexity', (m, perplexity_call.PerplexityModel.display_name(m))) for m in perplexity[:1]] +
                [('Open AI', (m, gpt_call.Model.display_name(m))) for m in gpt[1:]] +
                [('Google', (m, gemini_call.Model.display_name(m))) for m in gemini[1:]] +
                [('Meta', (m, together_llm_call.Model.display_name(m))) for m in llama[1:]] +
                [('Perplexity', (m, perplexity_call.PerplexityModel.display_name(m))) for m in perplexity[1:]]
                )


