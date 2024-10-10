# py-ship-simulator

`py-ship-simulator`は船舶操縦シミュレーションPythonライブラリです。



このライブラリの主な目的は以下の通りです。

- MMGモデルベースの操縦シミュレーションをより容易にすること
- 個人的なバグを最小限にすること

**デバッグは基本的に不十分ですから、十分に気をつけてください。また、バグ報告は必ずしてください。**



## Note

#### Memo

- [一次遅れモデル](./shipsim/ship/utils/response_models.py)のパラメータは適当に決定している。適宜修正してください
- 船のモデルは必要に応じて追加してください（参考：[EssoOsaka](./shipsim/ship/esso_osaka)）
- OpenAI gym環境は別レポジトリ（[environments-of-openai-gym-for-ship-maneuvering](https://github.com/NAOE-5thLab/environments-of-openai-gym-for-ship-maneuvering)）に存在します。



#### Development plan

- [ ] Logging Hydro-Force 
- [ ] Rendering tool & Real time plot





### Compilimg `.f90` files

本シミュレータはPythonのみで機能するが、FortranによりMMGモデルの高速化が可能である。しかし、F2pyの仕様上 `.f90` ファイルのコンパイルをあらかじめ行っておく必要があります。

Esso Osaka (3m)

```bash
gfortran -c shipsim/ship/esso_osaka/f2py_mmg/mmg_esso_osaka_verctor_input.f90
```

```bash
FC=gfortran f2py -m mmg_esso -c --f90flags='-O3' shipsim/ship/esso_osaka/f2py_mmg/mmg_esso_osaka_verctor_input.f90 --backend meson
```


## Requirement
- Python 3.9以下
- Numpy 2.0以上（f2pyコンパイルに必須）
- Pandas
- Matplotlib
- Scipy
- meson（f2pyコンパイルに必須）

