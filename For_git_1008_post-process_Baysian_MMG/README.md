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





## Usage

### installation

ソースコードをダウンロードした後、`shipsim`フォルダを作業ディレクトリに配置してください。



### Compilimg `.f90` files

本シミュレータはPythonのみで機能するが、FortranによりMMGモデルの高速化が可能である。しかし、F2pyの仕様上 `.f90` ファイルのコンパイルをあらかじめ行っておく必要があります。

Esso Osaka (3m)

```bash
gfortran -c shipsim/ship/esso_osaka/f2py_mmg/mmg_esso_osaka_verctor_input.f90
```

```bash
f2py --fcompiler=gnu95 -m mmg_esso -c --f90flags='-O3' shipsim/ship/esso_osaka/f2py_mmg/mmg_esso_osaka_verctor_input.f90
```



### Tutorial

[Zigzag test](./tutorial_zigzag.ipynb)と[DP test](./tutorial_vtps.ipynb)のノートブックを用意したので参考にしてください。





## Demo

##### Zigzag test of EssoOsaka in Inukai pond (Done by [tutorial_zigzag.ipynb](./tutorial_zigzag.ipynb))

<img src="./log/tutorial_zigzag/test_traj.png" style="zoom:30%;" />

##### Positioning test of Takaoki (Done by [tutorial_vtps.ipynb](./tutorial_vtps.ipynb))

<img src="./log/tutorial_vtps/test_traj.png" style="zoom:30%;" />



## Requirement
- Python 3.9以上（Genetic関数に対応したもの
- Numpy
- Pandas
- Matplotlib
- Scipy



## Update history

| Version             | Descriptions                                                 |
| ------------------- | ------------------------------------------------------------ |
| v0.1 (2023/Jul/27)  | 簡易的にシミュレータを作成した。                             |
| v1.0.0 (2023/Aug/2) | アクチュエータと船体で状態変数の数値積分方法が異なっていた構造問題を解決した。 <br />複数の船種と港湾を追加した。 |
| v1.1.0 (2023/Sep/4) | ShipとWorldの切り分け方法を整理し直した。                    |
| v1.1.1 (2023/Sep/9) | バグ修正（#6,#8,#9）                                         |
|                     |                                                              |

