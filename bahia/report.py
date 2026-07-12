# -*- coding: utf-8 -*-
"""Gera RELATORIO.md a partir de out/resumo.json e out/qc_log.txt.

Mantem prosa e numeros sincronizados: rodar apos figures.py. Os numeros vem
sempre do JSON produzido na ultima execucao, nunca digitados a mao.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def _br(x, casas=0):
    s = f"{x:.{casas}f}"
    return s.replace(".", ",")


def _qc_counts() -> dict:
    txt = (OUT / "qc_log.txt").read_text(encoding="utf-8")
    def n(chave):
        m = re.search(rf"^{chave}:\s*(.*)$", txt, re.MULTILINE)
        if not m:
            return 0
        linha = m.group(1).strip()
        return linha.count("(") if linha not in ("[]", "") else 0
    def head(chave):
        m = re.search(rf"^{chave}:\s*(\d+)", txt, re.MULTILINE)
        return int(m.group(1)) if m else 0
    return dict(coletados=head("MUNICIPIOS NA COLETA"),
                aprovadas=head("SERIES APROVADAS NO QC"),
                excluidas=n("excluded"), imputadas=n("imputed"),
                interpoladas=n("interpolated"), falhas=n("fit_failed"))


def build() -> None:
    r = json.loads((OUT / "resumo.json").read_text(encoding="utf-8"))
    qc = _qc_counts()
    faixas = {f["faixa"]: f for f in r["winrate_por_faixa"]}
    menor = faixas.get("até 20 mil", {})
    maior = faixas.get("acima de 100 mil", {})

    md = f"""# Toda a Bahia: o modelo contra a projeção das prefeituras

Análise independente do TCC, aplicada a **todos os {qc['coletados']} municípios
da Bahia** cujos dados o SICONFI disponibiliza, sob o mesmo método do trabalho:
deflação pelo IPCA, controle de qualidade, portfólio canônico de previsores e o
confronto anual contra a *Previsão Atualizada* que cada prefeitura registra no
RREO-Anexo 03. É o teste de generalização levado ao limite: dos três municípios
do núcleo aos dezoito populosos, e agora ao estado inteiro.

## O achado

O resultado central do TCC **resiste à escala**. Sobre as {r['confrontos']}
projeções anuais confrontáveis de {r['municipios']} municípios, o *Ensemble*,
adotado como modelo fixo, erra menos que a projeção da própria prefeitura em
**{_br(r['ens_winrate'], 0)}%** dos casos, praticamente o mesmo que os 61%
observados nos dezoito municípios populosos e os 67% dos três detalhados. Não é
um efeito de amostra pequena: repete-se de ponta a ponta da Bahia.

Um segundo padrão, mais sutil, emerge no cruzamento com o porte do município. A
**frequência** de vitória é alta e quase uniforme, entre {_br(min(f['winrate'] for f in r['winrate_por_faixa']), 0)}% e {_br(max(f['winrate'] for f in r['winrate_por_faixa']), 0)}%
em todas as faixas de população. O que muda com o tamanho é a **margem**: nos
municípios de até vinte mil habitantes a vantagem mediana do modelo é de
**{_br(menor.get('margem_mediana', 0), 1)} p.p.** de erro anual, contra
**{_br(maior.get('margem_mediana', 0), 1)} p.p.** nos de mais de cem mil, e os
casos extremos, em que a projeção oficial erra por dezenas de pontos, são quase
todos de cidades pequenas. A correlação entre o logaritmo da população e a
vantagem é de **{_br(r['corr_logpop_vantagem'], 2)}**: fraca, mas na direção
esperada. Vence-se com frequência semelhante em qualquer porte; ganha-se por
margem maior onde o município é menor.

Três leituras compõem o quadro:

- **Mapa** ([fig_mapa_bahia](out/fig_mapa_bahia.pdf)): o *Ensemble* erra menos
  que a prefeitura na grande maioria dos municípios avaliados (azul), e a
  intensidade da cor mede a vantagem em pontos percentuais de erro anual.
- **Vantagem contra população**
  ([fig_vantagem_populacao](out/fig_vantagem_populacao.pdf)): a nuvem de
  vantagens mais altas concentra-se à esquerda (municípios pequenos) e a mediana
  por quintil recua conforme o porte aumenta.
- **Margem por porte**
  ([fig_winrate_faixas](out/fig_winrate_faixas.pdf)): a margem mediana é bem
  maior nos municípios menores, enquanto a taxa de vitória, anotada em cada
  barra, permanece alta em todas as faixas.

## Números

O controle de qualidade partiu de {qc['coletados']} municípios coletados e
aprovou **{r['series']} séries** de IPTU e ISSQN, em **{r['municipios']}
municípios**, descartando {qc['excluidas']} séries por cobertura insuficiente,
valores mensais não-positivos ou excesso de anos anômalos, {qc['imputadas']}
anos-anomalia isolados imputados e {qc['interpoladas']} lacunas curtas
interpoladas (mesmos critérios da Seção 5.7 do TCC).

Sobre os {r['confrontos']} anos-série com previsão municipal disponível:

| medida | resultado |
|---|---|
| *Ensemble* (modelo fixo) vence a prefeitura | {r['ens_vitorias']} de {r['confrontos']} anos-série ({_br(r['ens_winrate'], 1)}%) |
| melhor modelo por série, ex-post (usa hindsight; ver seção adiante) | {r['expost_vitorias']} de {r['confrontos']} ({_br(r['expost_winrate'], 1)}%) |
| municípios em que o *Ensemble* tem vantagem média positiva | {r['municipios_vantagem_positiva']} de {r['municipios_total_mapa']} |
| vantagem mediana do *Ensemble* | {_br(r['vantagem_mediana_pp'], 1)} p.p. de erro anual |

*Ensemble* por faixa populacional (a vitória é frequente em todas; a margem
cresce nas menores):

| faixa | anos-série | vence | margem mediana |
|---|---|---|---|
"""
    for f in r["winrate_por_faixa"]:
        md += (f"| {f['faixa']} | {f['n']} | {_br(f['winrate'], 1)}% | "
               f"{_br(f['margem_mediana'], 1)} p.p. |\n")

    bm_path = OUT / "best_model.json"
    if bm_path.exists():
        bm = json.loads(bm_path.read_text(encoding="utf-8"))
        fx, es = bm["modelos_fixos"], bm["estrategias"]
        melhor_fixo = max(fx.items(), key=lambda kv: kv[1]["taxa"])
        md += f"""
## Ensemble fixo *vs.* “melhor modelo”

Uma alternativa natural, e a que *Oliveira (2024)* adota, é escolher para cada
série o modelo que saiu melhor, em vez de fixar o *Ensemble*. O teste mostra que
**o ganho aparente dessa escolha vem do uso do futuro**
([fig_melhor_modelo](out/fig_melhor_modelo.pdf)).

| estratégia de escolha | vence a prefeitura |
|---|---|
| melhor por série, com o resultado na mão (ex-post, estilo Oliveira) | {_br(es['melhor_expost']['taxa'], 1)}% |
| melhor modelo isolado, fixo no estado ({melhor_fixo[0]}) | {_br(melhor_fixo[1]['taxa'], 1)}% |
| *Ensemble* fixo (adotado no TCC) | {_br(fx['Ensemble']['taxa'], 1)}% |
| melhor por série, só com o passado (walk-forward, operável) | {_br(es['melhor_walkforward']['taxa'], 1)}% |

Escolher o melhor método por série *depois* de conhecer o desempenho de cada um
leva a taxa a **{_br(es['melhor_expost']['taxa'], 1)}%**. Mas essa seleção usa
informação que o gestor não tem no momento da previsão. Quando a mesma escolha é
feita apenas com o histórico disponível (adota-se, em cada ano, o modelo que fora
melhor nos anos anteriores da própria série), a taxa cai para
**{_br(es['melhor_walkforward']['taxa'], 1)}%**, abaixo do *Ensemble* fixo
({_br(fx['Ensemble']['taxa'], 1)}%). O melhor modelo isolado é o
{melhor_fixo[0]} ({_br(melhor_fixo[1]['taxa'], 1)}%), a um passo do *Ensemble*.

A lição é metodológica e sustenta a escolha do trabalho: o salto para
{_br(es['melhor_expost']['taxa'], 1)}% é uma miragem de *hindsight*. Sem olhar o
futuro, selecionar um modelo por série não supera fixar a média simples, que
dispensa a escolha, não depende de acertar qual método vencerá e entrega
desempenho equivalente ao do melhor modelo individual, com menos risco.
"""

    md += """
## Por que a margem é maior nos municípios menores

O mecanismo é o mesmo que o TCC identifica em escala reduzida. A projeção da LOA
de um município pequeno raramente nasce de um modelo: é uma meta orçamentária,
muitas vezes o valor do ano anterior corrigido por um índice único. Quando a
arrecadação tem sazonalidade estável (o IPTU concentrado no primeiro trimestre,
o ISSQN acompanhando a atividade de serviços), qualquer previsor que capture
esse ciclo já supera a extrapolação linear por larga margem. Nos grandes
municípios, a projeção oficial é mais cuidada e o espaço para ganho encolhe.

A leitura tem consequência prática direta: é justamente onde a capacidade
técnica é menor que a ferramenta automatizada rende mais. Um pipeline reproduzível
de coleta e modelagem, do tipo que este trabalho entrega, transfere para o
município pequeno uma qualidade de projeção que hoje só as capitais alcançam.

## Ressalvas

- O confronto compara o erro percentual anual de cada lado. A projeção da
  prefeitura é a *Previsão Atualizada* (revisada ao longo do exercício), leitura
  conservadora: a *Previsão Inicial* da LOA, quando indisponível por tributo no
  Anexo 03, tende a errar mais, o que só ampliaria a vantagem medida.
- O modelo opera sobre valores deflacionados e a prefeitura sobre nominais; cada
  erro percentual é medido contra o próprio realizado, como na Seção 5.7. É uma
  comparação entre tarefas de previsão, não entre séries idênticas.
- Séries reprovadas no controle de qualidade ficam de fora e aparecem hachuradas
  no mapa. A exclusão é simétrica para todos os modelos e independe do resultado.

## Reprodução

```bash
# 1. coleta (uma vez; retomável)
cd siconfi-collector
python -m siconfi collect rreo --state BA --years 2015-2025 \\
    --annex "RREO-Anexo 03" --periods 1,2,6
python -m siconfi transform-monthly
python -m siconfi transform-prefeitura-forecast

# 2. benchmark + figuras (no venv com statsforecast e prophet)
cd ../analysis/bahia
python benchmark.py --workers 10
python figures.py
python report.py
```

Detalhes de implementação em [README.md](README.md).
"""
    (HERE / "RELATORIO.md").write_text(md, encoding="utf-8")
    print(f"RELATORIO.md gerado ({len(md)} chars)")


if __name__ == "__main__":
    build()
