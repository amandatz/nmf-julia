# nmf-julia

Implementação em Julia de algoritmos para **Fatoração de Matrizes Não-Negativas** (NMF), com aplicação em reconhecimento de faces e comparação experimental entre os métodos.

---

## Descrição

Dado uma matriz não-negativa $X \in \mathbb{R}^{n \times m}_+$ e um inteiro positivo $r < \min(n, m)$, o problema de NMF consiste em encontrar matrizes não-negativas $W \in \mathbb{R}^{n \times r}_+$ e $H \in \mathbb{R}^{r \times m}_+$ tais que

$$X \approx WH,$$

minimizando a função objetivo

$$F(W, H) = \frac{1}{2} \|X - WH\|_F^2 \quad \text{sujeito a} \quad W \geq 0,\ H \geq 0.$$

---

## Algoritmos Implementados

### Atualizações Multiplicativas — `multiplicative.jl`

Baseado em Lee & Seung (2001). As regras de atualização têm a forma

$$H_{bj}^{k+1} = H_{bj}^k \cdot \frac{\left((W^k)^T V\right)_{bj}}{\left((W^k)^T W^k H^k\right)_{bj}}, \qquad W_{ia}^{k+1} = W_{ia}^k \cdot \frac{\left(V (H^{k+1})^T\right)_{ia}}{\left(W^k H^{k+1} (H^{k+1})^T\right)_{ia}}.$$

A função objetivo é monotonicamente não-crescente ao longo das iterações. A positividade de $W$ e $H$ é preservada desde que as inicializações sejam estritamente positivas e $V$ não possua colunas ou linhas nulas.

### Gradiente Projetado — `lin.jl`

Baseado em Lin (2007). Resolve o problema por mínimos quadrados não-negativos alternados com gradiente projetado O método possui convergência para ponto estacionário garantida teoricamente e converge empiricamente mais rápido que as atualizações multiplicativas.

---

## Estrutura do Repositório

```
nmf-julia/
├── data/
│   └── att_face_dataset/       # Dataset AT&T (ORL) de faces
├── resultados/                 # Saídas dos experimentos (figuras, métricas)
├── scripts/
│   ├── exact_x.jl             # Experimentos sintéticos com X exata
│   ├── face_recognition.jl    # Reconhecimento de faces com NMF
│   └── run_experiments.jl     # Experimentos sintéticos
├── src/
│   ├── algorithms/
│   │   ├── lin.jl             # Gradiente projetado (Lin, 2007)
│   │   ├── multiplicative.jl  # Atualizações multiplicativas (Lee & Seung, 2001)
│   ├── NMFProject.jl          # Módulo principal
│   ├── step_rules.jl          # Regras de passo e busca linear
│   └── utils.jl               # Funções auxiliares
├── Manifest.toml
└── Project.toml
```

---

## Instalação

Requer Julia ≥ 1.9.

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

---

## Uso

### Comparação dos algoritmos

```julia
include("scripts/run_experiments.jl")
```

```julia
include("scripts/exact_x.jl")
```

Compara os dois algoritmos em dados sintéticos.

### Reconhecimento de faces

```julia
include("scripts/face_recognition.jl")
```

Aplica NMF ao dataset AT&T Faces e avalia o desempenho de classificação com as bases $W$ aprendidas por cada algoritmo.

O dataset utilizado é o [AT&T Database of Faces (ORL)](https://cam-orl.co.uk/facedatabase.html), composto por 400 imagens em escala de cinza de 40 indivíduos (10 imagens por indivíduo), com dimensão $92 \times 112$ pixels.
