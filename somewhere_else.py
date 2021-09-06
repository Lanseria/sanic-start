from sanic import Sanic

app = Sanic.get_app("MyHelloWorldApp")


@app.post('/')
async def hello_post(request):
    if app.ctx.db is None:
        return text('none')
    else:
        return text(app.ctx.db)
