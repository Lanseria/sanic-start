from sanic import Sanic
from sanic.response import text, HTTPResponse
from sanic.request import Request

app = Sanic("MyHelloWorldApp")
db_settings = {
    'DB_HOST': 'localhost',
    'DB_NAME': 'appdb',
    'DB_USER': 'appuser'
}
app.config.update(db_settings)


@app.get("/")
async def hello_world(request: Request) -> HTTPResponse:

    app.ctx.db = "fack db"
    return text("Hello, world.")


@app.get("/foo")
async def foo_handler(request: Request) -> HTTPResponse:
    return text("I said foo!")


@app.get("/sync")
def sync_handler(request: Request) -> HTTPResponse:
    time.sleep(0.1)
    return text("Done.")


@app.get("/async")
async def async_handler(request: Request) -> HTTPResponse:
    await asyncio.sleep(0.1)
    return text("Done.")


@app.post('/request_json')
async def requestJson(request: Request) -> HTTPResponse:
    # print(request.json)
    # print(request.body)
    # print(request.form)
    print(request.files)
    # print(request.body.decode("UTF-8"))
    return text("request_json")
