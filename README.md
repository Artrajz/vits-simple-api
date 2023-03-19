# MoeGoe-Simple-API

Based on [MoeGoe](https://github.com/CjangCjengh/MoeGoe)

# How to use

1. Download VITS model and put it in folder *Model.*
2. Edit the model path in app.py.
3. Install requirements `pip install -r requirements.txt`
4. start！`python app.py`

## parameter

| Name         | Parameter | Is must | Default | Value     | Instruction                                           |
| ------------ | --------- | ------- | ------- | --------- | ----------------------------------------------------- |
| text         | text      | true    |         | text      |                                                       |
| speaker id   | id        | false   | 0       | (number)  |                                                       |
| audio format | format    | false   | wav     | wav,ogg   |                                                       |
| language     | lang      | false   | mix     | zh,ja,mix | texts should be wrapped by [ZH] or [JA] when lang=mix |

## merging

- GET http://127.0.0.1:23456/voice/speakers

  return speakers list (id and name)

```
[
	"0\t綾地寧々",
	"1\t在原七海",
	"2\t小茸",
	"3\t唐乐吟",
	"4\tルイズ",
	"5\tティファニア",
	"6\tイルククゥ",
	"7\tアンリエッタ",
	"8\tタバサ",
	"9\tシエスタ",
	"10\tハルナ",
	"11\t少女リシュ",
	"12\tリシュ",
	"13\tアキナ",
	"14\tクリス",
	"15\tカトレア",
	"16\tエレオノール",
	"17\tモンモランシー",
	"18\tリーヴル",
	"19\tキュルケ",
	"20\tウェザリー",
	"21\tサイト",
	"22\tギーシュ",
	"23\tコルベール",
	"24\tオスマン",
	"25\tデルフリンガー",
	"26\tテクスト",
	"27\tダンプリメ",
	"28\tガレット",
	"29\tスカロン"
]
```

- GET http://127.0.0.1/voice?text=[JA]text[JA][ZH]text[ZH]&id=0&format=wav&lang=mix

return wav audio file

- GET http://127.0.0.1/voice?text=[JA]text[JA][ZH]text[ZH]&id=0&format=ogg&lang=mix

return ogg audio file

- GET http://127.0.0.1/voice?text=[ZH]text&id=0&format=wav&lang=zh

send chinese and return  wav audio file



## Single language/model

### Japanese

- GET http://127.0.0.1/voice/ja?text=text&id=0&format=wav

return wav audio file

- GET http://127.0.0.1/voice/ja?text=text&id=0&format=ogg

return ogg audio file

### Chinese

- GET http://127.0.0.1/voice/zh?text=text&id=0&format=wav

return wav audio file

- GET http://127.0.0.1/voice/zh?text=text&id=0&format=ogg

return ogg audio file
