from fastapi import FastAPI, status
from dotenv import load_dotenv
import logfire
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# global variables. There should be a better approach than this.
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path="WeiboAI/VibeThinker-1.5B",
            dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path="WeiboAI/VibeThinker-1.5B"
        )
    except Exception as e:
        raise e
    yield
    # TODO: Not sure if model cleanup is required
    # import gc

    # del model
    # gc.collect()
    # torch.cuda.empty_cache()


app = FastAPI(
    title="FastAPI local llm model API",
    description="This is a FastAPI application serving calls to local llm model",
    version="0.1.0",
    lifespan=lifespan,
)


# View logs in Logfire
logfire.configure()
# exclude_urls takes comma separated regex that will not create logs for the matching pattern
logfire.instrument_fastapi(app, excluded_urls="docs$,openapi.json$")


@app.get("/health", status_code=status.HTTP_200_OK)
def get_health():
    global model
    global tokenizer
    model_input = tokenizer("Who are you?", return_tensors="pt").to(model.device)
    generate_ids = model.generate(**model_input)
    model_output = tokenizer.batch_decode(generate_ids, skip_special_token=True)[0]
    print(model_output)
    return {"message": f"Health ok {model_output}"}
