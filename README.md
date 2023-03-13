# MoeGoe-Simple-API

Based on [MoeGoe](https://github.com/CjangCjengh/MoeGoe)

# How to use

1. Download VITS model and put it in *Model.*
2. Modify the model path in app.py.
3. Install requirements and start.

```
pip install -r requirements.txt

python app.py
```

## Japanese

- GET http://127.0.0.1/api/ja?text=text&id=0&format=wav

return wav audio file

- GET http://127.0.0.1/api/ja?text=text&id=0&format=ogg

return ogg audio file

## Chinese

- GET http://127.0.0.1/api/zh?text=text&id=0&format=wav

return wav audio file

- GET http://127.0.0.1/api/zh?text=text&id=0&format=ogg

return ogg audio file
