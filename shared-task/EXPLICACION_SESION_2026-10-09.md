# Sesión del 9 de octubre de 2026: semillas, ensemble, máscara y heldout

Continuación de `EXPLICACION_SESION_2026-10-07.md`, con el mismo estilo: qué se
hizo, por qué se decidió cada cosa, qué salió y qué es **hecho medido** frente a
**hipótesis**. Los números son los medidos ese día. Las hipótesis se marcan
explícitamente como tales.

**Regla de reporte de esta sesión, sin excepciones.** Cada medición se da con
los cuatro números que promedia el ranking del concurso (WER y CER en scripted,
WER y CER en spontaneous) más su media. Nunca solo WER.

---

## Índice

0. [Resumen en una página](#0-resumen-en-una-página)
1. [El plan y por qué ese orden](#1-el-plan-y-por-qué-ese-orden)
2. [Paso 1: las semillas](#2-paso-1-las-semillas)
3. [Paso 2: el ensemble](#3-paso-2-el-ensemble)
4. [Paso 3: mask_time_prob](#4-paso-3-mask_time_prob)
5. [Paso 4: decisiones cerradas](#5-paso-4-decisiones-cerradas)
6. [Paso 5: el heldout](#6-paso-5-el-heldout)
7. [Paso 6: la entrega](#7-paso-6-la-entrega)
8. [Hechos frente a hipótesis](#8-hechos-frente-a-hipótesis)
9. [Archivos y cómo reproducir cada número](#9-archivos-y-cómo-reproducir-cada-número)
10. [Tabla final con todos los resultados](#10-tabla-final-con-todos-los-resultados)
11. [Glosario de esta sesión](#11-glosario-de-esta-sesión)
12. [Preguntas para profundizar con una IA](#12-preguntas-para-profundizar-con-una-ia)

---

## 0. Resumen en una página

**Lo que se hizo:** tres semillas más de la receta final; un ensemble (dos
formas de combinar); un barrido de `mask_time_prob`; decisiones cerradas solo
con dev; el heldout medido una sola vez, con pre-registro; y la entrega
regenerada y verificada.

**Las decisiones** (sección 5): **modelo único**, sin ensemble;
**`mask_time_prob` 0,05**; decodificación **v3**. Es decir, nada cambia respecto
a la entrega del día 8, y ahora está respaldado por medidas.

**Los resultados negativos**, reportados como tales:

- **El ensemble no gana.** Promediando logits empeora mucho (media WER/CER
  −1,36 / −0,17 frente a la media de semillas en el dev disjunto), por un motivo
  medido: CTC emite picos y las semillas los colocan a veces en frames
  contiguos, así que la media geométrica borra letras (borrados de carácter
  21 → 50). Promediando probabilidades ya no pierde letras, pero tampoco gana:
  en el mejor caso empata (±0,00).
- **Más máscara empeora.** 0,2 y 0,5 pierden los cuatro números.

**El heldout** (sección 6). Resultado principal, `both_n500_lr2e-5` + v3 en el
heldout disjunto en hablante y texto:

| | Scripted WER / CER | Spontaneous WER / CER | Media WER / CER |
|---|---|---|---|
| dev | 4,60 / 0,57 | 9,01 / 1,29 | 6,81 / 0,93 |
| **heldout** | **10,90 / 2,35** | **3,30 / 0,38** | **7,10 / 1,37** |

Las medias se parecen, pero esconden movimientos grandes y opuestos. Scripted
empeora mucho y spontaneous mejora mucho. Casi todo es **composición**, no ajuste
al dev:

- El heldout scripted contiene a `e990fbdf`, un hablante que por sí solo sube
  el WER scripted del dev de `sharedtask` de 5,71 a 9,23 (hecho medido; su
  efecto en el heldout es una inferencia, no una medida).
- El 46 % del heldout spontaneous es de hablantes cuya voz el modelo ya oyó en
  las transcripciones silver de train.

Lo que sí se puede afirmar con el heldout: la receta es estable (0,10 de rango
entre semillas); el orden de las semillas no se mantiene de dev a heldout, lo
que confirma que sus diferencias en dev eran ruido; y **la ganancia del LM en
spontaneous no aparece en el heldout** (neutra).

**La entrega:** `submission/final_n500_lr2e-5_v3_2026-10-09/qxp.zip`, verificada
e idéntica byte a byte a la del día 8.

---

## 1. El plan y por qué ese orden

Se partía de esta situación: el modelo de entrega (`final_n500_lr2e-5`, 500
pasos, LR 2e-5, semilla 42) es una sola muestra de una receta cuyo ruido entre
semillas es de ~0,30 de WER medio. **No sabíamos si ese modelo concreto había
tenido suerte.** Además, todas las cifras reportadas hasta entonces salían del
**dev**, que es el conjunto con el que se habían tomado todas las decisiones.

El plan tenía seis pasos, y el orden importa:

1. **Entrenar más semillas.** Sirven para dos cosas: medir la dispersión de
   verdad (no suponerla) y tener los miembros de un ensemble.
2. **Ensemble**: ¿promediar varios modelos gana más que el ruido entre semillas?
3. **mask_time_prob**: la regularización principal, que nunca se había tocado.
4. **Cerrar las decisiones.** A partir de aquí no se cambia nada.
5. **Heldout, una sola vez.** Es la única medición que no ha intervenido en
   ninguna decisión. Si después de mirarlo se cambiara algo, dejaría de ser
   heldout y se convertiría en otro dev.
6. **Generar la entrega.**

**Por qué el heldout va al final y solo una vez.** Cada vez que se mira un
conjunto de datos y se decide algo en función de lo que sale, ese conjunto deja
de ser una medida imparcial: el sistema se ha ajustado a él, aunque sea un poco.
Es lo que pasó con el dev a lo largo de todo el proyecto (pasos, LR, pesos del
LM, decodificación por dominio...). El heldout existe desde el primer día y
nunca se había usado, así que es la única estimación limpia que queda. Para
protegerlo:

- se fijaron todas las decisiones **antes** de medirlo;
- el script del heldout lleva un **pre-registro** en su cabecera (qué número es
  "el resultado", qué se mide además y qué no puede cambiar), y se guardó una
  copia con hora antes de ejecutarlo;
- el script **se niega a sobrescribir** un resultado de heldout ya existente;
- antes de ejecutarlo verifiqué que no existía ningún archivo de métricas ni
  ningún log de heldout en todo el proyecto: nunca se había medido.

**Una regla práctica que se respetó todo el día:** nunca ejecutar inferencia
mientras un entrenamiento ocupa la GPU (hallazgo K de HANDOFF: eso produjo una
vez un zip con 560 filas vacías). Todo se encadenó: primero los entrenamientos,
después las evaluaciones.

---

## 2. Paso 1: las semillas

### 2.1 Qué es una semilla y por qué hacen falta varias

La **semilla** fija el azar del entrenamiento: el orden en que se presentan los
clips y qué trozos del audio se enmascaran. Con la misma receta y otra semilla
sale un modelo algo distinto. Con un solo modelo no se puede saber si un buen
resultado es de la receta o de la suerte; con tres se puede estimar la
**dispersión** (cuánto varía el resultado por puro azar).

### 2.2 Qué se entrenó

De la receta ganadora (500 pasos, LR 2e-5, desde el baseline `ft_cpt_silver`)
ya existían la semilla 42 en los dos splits y la 43 en `sharedtask_both`.
Faltaban tres, ~7 minutos cada una:

| Modelo | Split de entrenamiento | Semilla |
|---|---|---|
| `checkpoints/hf/exp/both_n500_lr2e-5_s44` | `sharedtask_both` | 44 |
| `checkpoints/hf/final_n500_lr2e-5_s43` | `sharedtask` | 43 |
| `checkpoints/hf/final_n500_lr2e-5_s44` | `sharedtask` | 44 |

Recordatorio de los dos splits:

- **`sharedtask`**: disjunto solo por hablante. Las frases de dev están en train
  (leídas por otros), así que su dev scripted es **optimista** (hallazgo I: ~1,1
  puntos de WER). Es el split del paper original y con el que se entrena el
  modelo que se entrega.
- **`sharedtask_both`**: disjunto en hablante **y** en texto. Es el que se parece
  al test. Su dev scripted es pequeño (339 clips, 4 hablantes, 1.347 palabras).

### 2.3 Resultados por semilla (dev, decodificación v3)

"v3" es la decodificación del sistema real: scripted sin LM (greedy),
spontaneous con LM (`spont_all_o8`, α 0,5, β 0).

| Modelo | Split | scr WER | scr CER | spo WER | spo CER | media WER | media CER |
|---|---|---:|---:|---:|---:|---:|---:|
| semilla 42 | both | 4,60 | 0,57 | 9,01 | 1,29 | 6,81 | 0,93 |
| semilla 43 | both | 4,97 | 0,64 | 9,37 | 1,33 | 7,17 | 0,99 |
| semilla 44 | both | 4,97 | 0,63 | 9,20 | 1,32 | 7,09 | 0,97 |
| **media ± desv.** | both | 4,85 ± 0,21 | 0,62 ± 0,04 | 9,20 ± 0,18 | 1,31 ± 0,02 | **7,02 ± 0,19** | **0,96 ± 0,03** |
| semilla 42 | sharedtask | 9,23 | 3,01 | 9,06 | 1,30 | 9,14 | 2,16 |
| semilla 43 | sharedtask | 9,46 | 2,99 | 9,08 | 1,28 | 9,27 | 2,13 |
| semilla 44 | sharedtask | 9,28 | 2,92 | 9,28 | 1,32 | 9,28 | 2,12 |
| **media ± desv.** | sharedtask | 9,32 ± 0,12 | 2,97 ± 0,05 | 9,14 ± 0,12 | 1,30 ± 0,02 | **9,23 ± 0,08** | **2,14 ± 0,02** |

(± es la desviación estándar muestral de las tres semillas.)

**Cómo leerlo:**

- **La dispersión medida es la que se suponía.** En `sharedtask_both`, el
  rango (máximo menos mínimo) del WER medio es 0,37 y la desviación 0,19. En
  `sharedtask` el rango es menor (0,13), probablemente porque su dev es más
  grande (1.493 clips scripted frente a 339).
- **La semilla 42 de `sharedtask_both` fue la afortunada**: es la mejor de las
  tres en los cuatro números. Ya se sospechaba en la sesión anterior.
- **El modelo de entrega (semilla 42 en `sharedtask`) no tuvo una suerte
  especial**: está en la media (9,14 frente a 9,23 ± 0,08).

---

## 3. Paso 2: el ensemble

### 3.1 La idea

Un **ensemble** combina varios modelos para que sus errores aleatorios se
compensen. Aquí: pasar el audio por los tres modelos, combinar sus salidas frame
a frame y decodificar esa combinación una sola vez. Si el ruido entre semillas
es real, combinar tres semillas debería reducirlo y dejar un sistema mejor que
la media de los tres.

**El criterio de decisión, fijado antes de medir:** el ensemble solo merece la
pena si gana **más que la dispersión entre semillas (~0,30)** y lo hace **en WER
y CER a la vez**.

### 3.2 La implementación

`shared-task/ensemble.py` define `CTCEnsemble`, que se comporta como un único
modelo CTC: su `forward` devuelve la combinación de las salidas de los
miembros, y la conversión de longitudes (que se usa para recortar el relleno de
cada clip) la delega en el primer miembro. Gracias a eso, todo lo que ya
existía funciona sin cambios sobre él: la decodificación greedy de
`04_evaluate.py`, `ctc_logprobs` y el decodificador con LM de `ctc_lm.py`.
`04_evaluate.py` y `05_predict.py` aceptan ahora varios `--model`. Con uno solo,
el camino es exactamente el de siempre.

**Comprobaciones de seguridad.** Promediar frames solo tiene sentido si los
frames son "los mismos" en todos los modelos. `load_ctc` exige que coincidan el
vocabulario (el mismo símbolo en el mismo índice), el preprocesado del audio y
el frontend convolucional (la misma tasa de frames). Si algo no coincide, se
niega a cargarlos.

**Prueba unitaria** (en CPU, con modelos diminutos aleatorios, sin tocar la GPU):
con un solo modelo se devuelve el modelo tal cual; el ensemble de un modelo
consigo mismo da sus mismas salidas; con tres modelos distintos la combinación
es exactamente la media pedida; las dos rutas de decodificación funcionan; y un
modelo con el vocabulario permutado se rechaza.

### 3.3 Primer intento: media de logits. Pierde mucho

La especificación decía "promediar sus logits por frame". Eso es lo que se
implementó primero:

| Sistema (dev, v3) | Split | scr WER | scr CER | spo WER | spo CER | media WER | media CER |
|---|---|---:|---:|---:|---:|---:|---:|
| media de las 3 semillas | both | 4,85 | 0,62 | 9,20 | 1,31 | 7,02 | 0,96 |
| **ensemble de logits** | both | 6,53 | 0,84 | 10,23 | 1,44 | **8,38** | **1,14** |
| media de las 3 semillas | sharedtask | 9,32 | 2,97 | 9,14 | 1,30 | 9,23 | 2,14 |
| **ensemble de logits** | sharedtask | 11,57 | 3,36 | 9,97 | 1,39 | **10,77** | **2,37** |

**Peor que cualquier semilla individual, en los cuatro números y en los dos
splits.** Un ensemble de modelos que parten del mismo punto no suele empeorar
1,4 puntos, así que lo primero era descartar un error mío.

### 3.4 El diagnóstico: no es un bug, es cómo funciona CTC

Sobre los 339 clips scripted del dev de `sharedtask_both`, con modelos reales:

1. **El ensemble de la semilla 42 consigo misma reproduce la semilla 42 en 339
   de 339 clips.** El código hace lo que debe.
2. **Qué tipo de error añade.** Por palabras salían "sustituciones", pero una
   letra perdida dentro de una palabra cuenta como sustitución de la palabra
   entera. Por caracteres se ve el mecanismo:

   | | Sustituciones | **Borrados** | Inserciones |
   |---|---:|---:|---:|
   | semilla 42 | 33 | 21 | 11 |
   | ensemble de logits | 35 | **50** | 10 |
   | ensemble de probabilidades | 35 | 23 | 10 |

   La media de logits **duplica con creces los borrados de caracteres** y deja
   igual lo demás.

3. **Por qué.** La salida de un modelo CTC son **picos**: la mayoría de los
   frames dicen "blank" con mucha seguridad, y cada letra aparece como un pico
   en uno o dos frames. Medido entre las semillas 42 y 43: el 85,5 % de los
   picos caen en el mismo frame, y el **14,4 % caen desplazados un frame**.

   Promediar logits equivale a la **media geométrica** de las probabilidades
   (multiplicarlas y sacar la raíz). Un ejemplo con números redondos y tres
   modelos, como el ensemble real: A y B ponen el pico de una letra `q` en el
   frame *t*, y C lo pone un frame después, en *t+1*:

   | | frame *t* | frame *t+1* |
   |---|---|---|
   | A | q 0,90 · blank 0,09 | q 0,005 · blank 0,99 |
   | B | q 0,90 · blank 0,09 | q 0,005 · blank 0,99 |
   | C | q 0,005 · blank 0,99 | q 0,90 · blank 0,09 |
   | **media geométrica** | q 0,16 · blank **0,20** → blank | q 0,03 · blank **0,45** → blank |
   | **media aritmética** | q **0,60** · blank 0,39 → **q** | q 0,30 · blank **0,69** → blank |

   Con la media geométrica, la letra **desaparece**: blank gana en los dos
   frames. Basta con que **un** modelo esté casi seguro de "blank" en un frame
   para hundir la probabilidad de `q` ahí, porque multiplicar por 0,005 lo
   arrastra todo. Funciona como un veto: todos tienen que estar de acuerdo en el
   mismo frame.

   Con la media aritmética, la letra **sobrevive** en el frame *t*, porque sigue
   a la mayoría: un modelo discrepante solo resta un tercio. Es lo que se ve en
   los datos reales: los borrados pasan de 50 a 23, casi lo mismo que el modelo
   solo (21).

   Es un problema conocido de fusionar modelos CTC frame a frame: sus
   alineaciones (en qué frame pone cada uno cada letra) no están sincronizadas.

### 3.5 Segundo intento: media de probabilidades. Tampoco gana

Para que el resultado negativo fuera justo, había que medir también la fusión
correcta. Se añadió `--ensemble_fusion prob`, que pasó a ser la opción por
defecto:

| Ensemble vs media de semillas (+ = ensemble mejor) | scr WER | scr CER | spo WER | spo CER | media WER | media CER |
|---|---:|---:|---:|---:|---:|---:|
| prob, `both`, v3 | +0,17 | +0,02 | −0,73 | −0,12 | −0,28 | −0,05 |
| prob, `both`, greedy | +0,17 | +0,02 | −0,17 | −0,02 | **±0,00** | **±0,00** |
| prob, `sharedtask`, v3 | −0,93 | −0,18 | −0,74 | −0,12 | −0,57 | −0,10 |
| prob, `sharedtask`, greedy | −0,93 | −0,18 | −0,22 | −0,03 | −0,57 | −0,10 |
| *(dispersión entre semillas, `both`)* | *0,37* | *0,07* | *0,36* | *0,04* | *0,37* | *0,06* |

En el mejor caso (`both`, greedy) **empata** con la media de las semillas; en
los demás, empeora. En ningún caso supera la dispersión ni mejora WER y CER a
la vez.

**Decisión: no hay ensemble. Se entrega el modelo único.** Es un **resultado
negativo** y queda reportado como tal.

### 3.6 Lo que no sé explicar del todo (hipótesis)

- **Por qué ni siquiera la media de probabilidades ayuda.** Hipótesis: los
  miembros se parecen demasiado (coinciden en el argmax del 96,6 % de los
  frames). Han salido del mismo modelo base y han entrenado solo 500 pasos con
  un LR bajo, así que sus errores están muy correlacionados, y promediar errores
  correlacionados no los cancela.
- **Por qué con el LM el ensemble de probabilidades empeora spontaneous**
  (9,92 frente a 9,56 en greedy), cuando con cualquier modelo individual el LM
  lo mejora. Hipótesis: la media aritmética reparte cada pico entre dos frames y
  deja posteriores menos marcados, de modo que α 0,5, ajustado para modelos
  individuales, le da al LM demasiado peso relativo.
- **Por qué el ensemble de probabilidades pierde 0,93 en el scripted de
  `sharedtask` y gana 0,17 en el de `sharedtask_both`.** No lo sé. El dev de
  `sharedtask` incluye al hablante `e990fbdf` (106 % de WER), donde los modelos
  divergen mucho; es una posible explicación, no comprobada.

---

## 4. Paso 3: mask_time_prob

### 4.1 Qué es

Durante el entrenamiento, wav2vec 2.0 aplica una forma de **SpecAugment**: tapa
tramos aleatorios del audio (en el espacio de las representaciones internas)
para que el modelo no dependa de ningún trozo concreto y aprenda a usar el
contexto. Es una regularización: hace el entrenamiento más difícil a propósito,
para que el modelo generalice mejor.

`mask_time_prob` controla **cuánto** se tapa. En la implementación de
HuggingFace se colocan aproximadamente `mask_time_prob × frames / 10` tramos
de 10 frames (200 ms) cada uno. Con 0,05 se tapa ~5 % de los frames; con 0,5,
hasta ~50 % (algo menos porque los tramos se solapan).

Nunca se había tocado: estaba en 0,05, el valor por defecto de HuggingFace. Las
recetas de fairseq para wav2vec 2.0 suelen usar valores mucho mayores (~0,5–0,75).
Como el problema observado era el sobreajuste, era el candidato natural.

**Comprobación previa.** `mask_time_prob` solo actúa si `apply_spec_augment`
está activado en la configuración del modelo. Lo verifiqué
(`apply_spec_augment: True`, `mask_time_length: 10`). También verifiqué que el
modelo entrenado con 0,5 tiene `mask_time_prob: 0.5` en su config.

### 4.2 Resultados

Semilla 42 en `sharedtask_both`, para compararlo emparejado con el run que ya
existía (misma semilla, solo cambia la máscara). Dev, decodificación v3:

| mask_time_prob | scr WER | scr CER | spo WER | spo CER | media WER | media CER |
|---|---:|---:|---:|---:|---:|---:|
| 0,05 (sin cambio) | 4,60 | 0,57 | 9,01 | 1,29 | 6,81 | 0,93 |
| 0,2 | 5,05 | 0,66 | 9,06 | 1,30 | 7,05 | 0,98 |
| 0,5 | 5,79 | 0,76 | 9,56 | 1,38 | 7,68 | 1,07 |

**Las dos empeoran los cuatro números, y más cuanto mayor es la máscara.** La
regla era repetir con la semilla 43 solo si algún valor ganaba más de 0,30
antes de creérselo; como ninguno gana, no hizo falta.

Un matiz: la semilla 42 fue la afortunada en 0,05 (sección 2.3), así que
compararse con ella es exigente. Pero incluso frente a la media de las tres
semillas en 0,05 (7,02 / 0,96), 0,2 queda peor (7,05 / 0,98) y 0,5 mucho peor.

**Decisión: `mask_time_prob` se queda en 0,05.** Otro resultado negativo.

### 4.3 Por qué podría pasar (hipótesis)

El sobreajuste que motivó probar la máscara (hallazgo C) apareció después de
~500 pasos en un run con un LR máximo 2,5 veces mayor (5e-5). La receta actual
para en 500 pasos con LR 2e-5: probablemente **nunca llega a la zona de
sobreajuste**, así que la regularización extra solo quita señal. Encaja con que
el daño crezca con la máscara. Para que una máscara mayor ayudara habría que
entrenar más tiempo, y la meseta del hallazgo H dice que más tiempo no aporta.

---

## 5. Paso 4: decisiones cerradas

Con lo medido en los pasos 2 y 3, y solo en dev:

| Decisión | Elegido | Por qué |
|---|---|---|
| ¿Ensemble o modelo único? | **Modelo único** | Ninguna fusión gana más que la dispersión, ni en WER ni en CER. |
| ¿Qué modelo único? | **`final_n500_lr2e-5` (semilla 42)** | Era ya la entrega. Elegir "la mejor semilla" según dev sería ajustar al ruido: las tres están a 0,13 de WER medio. |
| `mask_time_prob` | **0,05** | 0,2 y 0,5 empeoran los cuatro números. |
| Decodificación | **v3** (scripted greedy, spontaneous con LM) | Sin cambios respecto a la sesión anterior. |

**A partir de aquí no se cambió nada.**

---

## 6. Paso 5: el heldout

### 6.1 Qué es el heldout, exactamente

Dos heldouts, uno por split:

- **`sharedtask_both`, heldout scripted:** 364 clips, 7 hablantes, ninguno en
  train, y ninguna frase en train. Es la medida más limpia de scripted.
- **`sharedtask`, heldout scripted:** 1.328 clips, 3 hablantes que no están en
  train, pero **todas sus frases sí** (diseño del paper original). Optimista,
  como su dev.
- **Heldout spontaneous**, el mismo en los dos splits: 279 clips, los que los
  organizadores marcaron como `test` en el corpus spontaneous. Nunca se han
  usado para entrenar.

### 6.2 El pre-registro

Escrito en la cabecera de `experiments/2026-10-09_eval_heldout.sh` antes de
ejecutarlo, con una copia fechada en `results/2026-10-09/PREREGISTRATION_heldout_145955.sh`:

- **Principal ("el resultado del paper"):** `both_n500_lr2e-5` (semilla 42)
  + v3 en el heldout de `sharedtask_both`.
- **Entrega:** `final_n500_lr2e-5` (semilla 42) + v3 en el heldout de `sharedtask`.
- **Secundarias, no deciden nada:** las semillas 43 y 44 de las dos recetas
  (dispersión en heldout) y la decodificación greedy de cada modelo (efecto
  del LM en heldout).
- **Nada cambia después de ver estos números.**

Se ejecutó una vez (14:59–15:11). Ningún número de esta sección se ha usado para
cambiar nada.

### 6.3 Resultados (decodificación v3), junto a los de dev del mismo modelo

| Modelo | Conjunto | scr WER | scr CER | spo WER | spo CER | media WER | media CER |
|---|---|---:|---:|---:|---:|---:|---:|
| **both s42 (principal)** | dev | 4,60 | 0,57 | 9,01 | 1,29 | 6,81 | 0,93 |
| **both s42 (principal)** | **heldout** | **10,90** | **2,35** | **3,30** | **0,38** | **7,10** | **1,37** |
| both s43 | dev | 4,97 | 0,64 | 9,37 | 1,33 | 7,17 | 0,99 |
| both s43 | heldout | 10,62 | 2,32 | 3,42 | 0,40 | 7,02 | 1,36 |
| both s44 | dev | 4,97 | 0,63 | 9,20 | 1,32 | 7,09 | 0,97 |
| both s44 | heldout | 10,83 | 2,32 | 3,42 | 0,39 | 7,13 | 1,36 |
| both, media ± desv. | dev | 4,85 ± 0,21 | 0,62 ± 0,04 | 9,20 ± 0,18 | 1,31 ± 0,02 | 7,02 ± 0,19 | 0,96 ± 0,03 |
| both, media ± desv. | heldout | 10,79 ± 0,14 | 2,33 ± 0,02 | 3,38 ± 0,07 | 0,39 ± 0,01 | 7,08 ± 0,05 | 1,36 ± 0,01 |
| **sharedtask s42 (entrega)** | dev | 9,23 | 3,01 | 9,06 | 1,30 | 9,14 | 2,16 |
| **sharedtask s42 (entrega)** | **heldout** | **9,71** | **1,76** | **3,62** | **0,43** | **6,66** | **1,09** |
| sharedtask s43 | dev | 9,46 | 2,99 | 9,08 | 1,28 | 9,27 | 2,13 |
| sharedtask s43 | heldout | 9,62 | 1,77 | 3,44 | 0,40 | 6,53 | 1,08 |
| sharedtask s44 | dev | 9,28 | 2,92 | 9,28 | 1,32 | 9,28 | 2,12 |
| sharedtask s44 | heldout | 9,78 | 1,78 | 3,62 | 0,42 | 6,70 | 1,10 |
| sharedtask, media ± desv. | dev | 9,32 ± 0,12 | 2,97 ± 0,05 | 9,14 ± 0,12 | 1,30 ± 0,02 | 9,23 ± 0,08 | 2,14 ± 0,02 |
| sharedtask, media ± desv. | heldout | 9,70 ± 0,08 | 1,77 ± 0,01 | 3,56 ± 0,10 | 0,41 ± 0,02 | 6,63 ± 0,09 | 1,09 ± 0,01 |

**De dev a heldout, en el modelo principal:** scripted empeora +6,30 de WER y
+1,78 de CER; spontaneous mejora −5,71 y −0,91; la media sube +0,29 / +0,44.
**En el de entrega:** scripted +0,48 de WER pero −1,25 de CER; spontaneous
−5,44 / −0,87; la media baja −2,48 / −1,07.

Leído solo por la media, el modelo principal parecería "casi igual en
heldout que en dev". **Sería engañoso:** los dos dominios se mueven 6 puntos en
sentidos opuestos y se compensan por casualidad. Por eso la regla de reportar
siempre los cuatro números.

### 6.4 Por qué dev y heldout difieren tanto: composición, no (solo) ajuste

Antes de atribuir la diferencia a haber ajustado el sistema al dev, comprobé
la composición de los conjuntos **solo con metadatos**, sin volver a pasar
ningún modelo por el heldout:

| Conjunto scripted | Clips | Hablantes | ¿Contiene `e990fbdf`? | Mayor hablante |
|---|---:|---:|---|---:|
| `both` dev | 339 | 4 | no | 73 % de los clips |
| `both` heldout | 364 | 7 | **sí** | 34 % |
| `sharedtask` dev | 1.493 | 7 | **sí** | 51 % |
| `sharedtask` heldout | 1.328 | 3 | no | 56 % |

Y lo que pesa **un solo hablante**, medido en dev (se puede, porque el dev no
está protegido), con el modelo de entrega:

| Dev scripted de `sharedtask` | Clips | WER | CER |
|---|---:|---:|---:|
| los 7 hablantes | 1.493 | 9,23 | 3,01 |
| sin `e990fbdf` | 1.434 | **5,71** | **0,93** |
| solo `e990fbdf` | 59 | 98,67 | 54,06 |

Con el 4 % de los clips, ese hablante sube el WER scripted 3,5 puntos y triplica
el CER. Está en el heldout de `both` y no en su dev, lo que **muy
probablemente** explica la mayor parte del salto de 4,60 a 10,90 de scripted.
Digo "muy probablemente" porque verificarlo exige volver a pasar los modelos por
el heldout guardando las predicciones, y no lo he hecho: aunque no cambiaría
ninguna decisión, sería volver a mirar el heldout.

En spontaneous, la sorpresa es la contraria: el heldout es **más fácil** (3,3
frente a 9,0). Dos hechos medidos lo explican en parte:

- **El 46 % de los clips del heldout spontaneous son de 2 hablantes que
  tienen 82 clips en train** (74 silver y 8 pendientes). Las transcripciones
  silver se repartieron sin mirar el hablante, así que **el heldout spontaneous
  no es disjunto por hablante**: el modelo ya había oído esas voces. En el dev
  esto afecta solo al 1 % de los clips.
- Sus hablantes tienen veintitantos y treinta y tantos años, igual que los del
  dev; ninguno se parece a los mayores de 60 del test.

**Conclusión honesta:** con conjuntos de 3 a 7 hablantes, **quién habla pesa más
que cualquier ajuste al dev**, y no se puede leer directamente "cuánto nos
ajustamos al dev" en la diferencia entre dev y heldout.

### 6.5 Lo que el heldout sí permite afirmar

1. **La receta es estable.** El rango entre semillas en heldout es 0,10 de WER
   medio en `both` y 0,17 en `sharedtask`: menor que en dev.
2. **Las diferencias entre semillas en dev eran ruido.** El orden no se
   mantiene: en el dev de `both` la mejor semilla era la 42 (6,81), y en heldout
   la mejor es la 43 (7,02) y la 42 queda en medio (7,10). Lo mismo en
   `sharedtask`. Esto confirma que no había que elegir "la mejor semilla"
   según dev.
3. **La ganancia del LM en spontaneous no se confirma.** En dev, el LM mejoraba
   spontaneous en las seis semillas (entre +0,12 y +0,65 de WER). En heldout
   es neutro: de media sale 3,47 con LM frente a 3,44 sin LM (−0,03), con el
   mismo CER (0,40). Es la huella típica del sesgo de selección: el LM, su peso
   y su texto se eligieron mirando el dev de spontaneous. También influye que
   el heldout spontaneous es mucho más fácil (3,4 % de WER) y deja poco que
   corregir. **No se cambia nada** (pre-registro), pero el paper debe
   presentar el LM como neutro, no como una mejora.
4. **El número principal para el paper**, con sus salvedades: heldout scripted
   **10,90 / 2,35** (frases y hablantes nuevos, incluido el hablante más
   difícil del corpus) y heldout spontaneous **3,30 / 0,38** (optimista: el 46 %
   de sus clips son de voces oídas en el silver de train).

---

## 7. Paso 6: la entrega

**El sistema:** el fijado en el paso 4. Modelo único `final_n500_lr2e-5`
(entrenado en `sharedtask`, semilla 42) con la decodificación v3 (scripted sin
LM, spontaneous con `spont_all_o8`, α 0,5, β 0). Comando en
`experiments/2026-10-09_submission.sh`.

**Resultado:** `submission/final_n500_lr2e-5_v3_2026-10-09/qxp.zip`.

**Verificación**, hecha por separado y no a partir del resumen del script:

| Comprobación | Resultado |
|---|---|
| Filas | 659, igual que la plantilla |
| Campos por fila | exactamente 2 en todas |
| Transcripciones vacías | 0 |
| Nombres y orden | idénticos a `data/test/qxp_test_dataset/qxp.tsv` |
| Nombres duplicados | 0 |
| Caracteres fuera de `a–z ñ ' espacio` | ninguno |
| Contenido del zip | solo `qxp.tsv`, idéntico al TSV en disco |
| Fin de línea | `\n`, sin `\r` |
| Frente a la entrega del 8 de octubre | **idéntica byte a byte** |

Que sea idéntica a la del día 8 es lo esperado, porque el sistema no cambió.
Además confirma que la inferencia es determinista: regenerar da exactamente el
mismo archivo. No está subida.

---

## 8. Hechos frente a hipótesis

**Hechos medidos:**

- La dispersión entre tres semillas de la receta final es de 0,37 de WER medio
  (rango) en el dev de `sharedtask_both` y de 0,13 en el de `sharedtask`.
- El ensemble por media de logits empeora los cuatro números en los dos splits.
  No es un bug: el ensemble de un modelo consigo mismo lo reproduce en 339 de
  339 clips.
- El mecanismo de esa pérdida son los borrados de caracteres (21 → 50).
- Los picos CTC de dos semillas coinciden en el mismo frame el 85,5 % de las
  veces y están desplazados un frame el 14,4 %; los argmax coinciden en el
  96,6 % de los frames.
- El ensemble por media de probabilidades no pierde caracteres (23 borrados),
  pero no gana más que la dispersión en ninguna configuración.
- `mask_time_prob` 0,2 y 0,5 empeoran los cuatro números frente a 0,05, con la
  misma semilla.
- El heldout no se había medido nunca antes de este día (comprobado buscando
  cualquier archivo de métricas o log de heldout).
- En heldout, la dispersión entre semillas es de 0,10 (`both`) y 0,17
  (`sharedtask`) de WER medio, y el orden de las semillas no coincide con el de dev.
- La ganancia del LM en spontaneous de dev (+0,12 a +0,65 de WER según la
  semilla) no aparece en heldout (−0,03 de media; CER igual).
- El hablante `e990fbdf` (59 clips) sube el WER scripted del dev de `sharedtask`
  de 5,71 a 9,23 y el CER de 0,93 a 3,01. Está en el heldout de `both` y no en
  su dev.
- El 46 % de los clips del heldout spontaneous son de hablantes con clips silver
  en train.
- La entrega regenerada es idéntica byte a byte a la del día 8.

**Hipótesis (no comprobadas):**

- Que el ensemble de probabilidades no gane porque los miembros están demasiado
  correlacionados.
- Que con el LM el ensemble de probabilidades empeore spontaneous porque sus
  posteriores son menos marcados y α 0,5 le da demasiado peso al LM.
- Que la diferencia del ensemble de probabilidades entre los dos devs scripted
  (+0,17 frente a −0,93) venga del hablante `e990fbdf`.
- Que la máscara no ayude porque la receta de 500 pasos no llega a sobreajustar.
- Que el salto de scripted de dev a heldout en `both` (4,60 → 10,90) se deba
  sobre todo a `e990fbdf`. Es muy probable, pero no está medido: haría falta
  pasar de nuevo los modelos por el heldout guardando predicciones.
- Que la ganancia del LM en dev fuera sesgo de selección y no una mejora real.
  También puede deberse a que el heldout spontaneous es mucho más fácil.

---

## 9. Archivos y cómo reproducir cada número

| Archivo | Qué es |
|---|---|
| `shared-task/ensemble.py` | **Nuevo.** `CTCEnsemble` y `load_ctc`: fusión `prob` (por defecto) o `logit`, con las comprobaciones de compatibilidad. |
| `shared-task/04_evaluate.py` | Acepta varios `--model`, `--ensemble_fusion` y `--metrics_json`; cada JSON guarda en `meta` qué modelos, split y decodificación se usaron. |
| `shared-task/05_predict.py` | Acepta varios `--model` y `--ensemble_fusion`. |
| `shared-task/experiments/2026-10-09_seeds_and_masking.sh` | Los cinco entrenamientos del día (pasos 1 y 3). |
| `shared-task/experiments/2026-10-09_eval_dev.sh` | Todas las mediciones en dev (pasos 2 y 3). |
| `shared-task/experiments/2026-10-09_eval_heldout.sh` | El heldout, con el pre-registro en la cabecera. Se niega a sobrescribir. |
| `shared-task/experiments/collect_2026-10-09.py` | Reúne los JSON en tablas: cuatro números + medias, dispersión y deltas. |
| `shared-task/experiments/2026-10-09_submission.sh` | El comando exacto de la entrega del paso 6. |
| `results/2026-10-09/` | Un JSON y un log por medición (`<split>__<sistema>__<decodificación>`), y la copia del pre-registro. **No va a git** (`results/*` está en el `.gitignore`). |

Para regenerar todas las tablas de este documento:

```bash
python shared-task/experiments/collect_2026-10-09.py --markdown
```

El análisis por hablante del anexo B usa las predicciones guardadas en
`results/2026-10-09/pred_dev_st_s42_v3/`.

Modelos nuevos (tampoco en git): `checkpoints/hf/exp/both_n500_lr2e-5_s44`,
`..._mask0.2`, `..._mask0.5`, `checkpoints/hf/final_n500_lr2e-5_s43` y `_s44`.

---

## 10. Tabla final con todos los resultados

Una fila por modelo y configuración. Todas las cifras en %. "Influyó" dice si
ese número intervino en alguna decisión. Decodificación v3 (la del sistema)
salvo donde se indica.

| Modelo / configuración | scr WER | scr CER | spo WER | spo CER | media WER | media CER | Split (conjunto) | ¿Influyó en una decisión? |
|---|---:|---:|---:|---:|---:|---:|---|---|
| both s42, mask 0,05 | 4,60 | 0,57 | 9,01 | 1,29 | 6,81 | 0,93 | sharedtask_both (dev) | sí: pasos 2 (dispersión) y 3 (referencia emparejada) |
| both s43 | 4,97 | 0,64 | 9,37 | 1,33 | 7,17 | 0,99 | sharedtask_both (dev) | sí: paso 2 (dispersión) |
| both s44 | 4,97 | 0,63 | 9,20 | 1,32 | 7,09 | 0,97 | sharedtask_both (dev) | sí: paso 2 (dispersión) |
| both, media de 3 semillas | 4,85 | 0,62 | 9,20 | 1,31 | 7,02 | 0,96 | sharedtask_both (dev) | sí: paso 2 (referencia del ensemble) |
| both, desv. estándar (3 semillas) | 0,21 | 0,04 | 0,18 | 0,02 | 0,19 | 0,03 | sharedtask_both (dev) | sí: paso 2 (umbral) |
| both, rango (máx − mín) | 0,37 | 0,07 | 0,36 | 0,04 | 0,37 | 0,06 | sharedtask_both (dev) | sí: paso 2 (umbral) |
| both, ensemble de logits | 6,53 | 0,84 | 10,23 | 1,44 | 8,38 | 1,14 | sharedtask_both (dev) | sí: paso 2 (descartado) |
| both, ensemble de probabilidades | 4,68 | 0,60 | 9,92 | 1,43 | 7,30 | 1,02 | sharedtask_both (dev) | sí: paso 2 (descartado) |
| both s42, mask 0,2 | 5,05 | 0,66 | 9,06 | 1,30 | 7,05 | 0,98 | sharedtask_both (dev) | sí: paso 3 (descartado) |
| both s42, mask 0,5 | 5,79 | 0,76 | 9,56 | 1,38 | 7,68 | 1,07 | sharedtask_both (dev) | sí: paso 3 (descartado) |
| sharedtask s42 (entrega) | 9,23 | 3,01 | 9,06 | 1,30 | 9,14 | 2,16 | sharedtask (dev) | sí: paso 2 (dispersión) |
| sharedtask s43 | 9,46 | 2,99 | 9,08 | 1,28 | 9,27 | 2,13 | sharedtask (dev) | sí: paso 2 (dispersión) |
| sharedtask s44 | 9,28 | 2,92 | 9,28 | 1,32 | 9,28 | 2,12 | sharedtask (dev) | sí: paso 2 (dispersión) |
| sharedtask, media de 3 semillas | 9,32 | 2,97 | 9,14 | 1,30 | 9,23 | 2,14 | sharedtask (dev) | sí: paso 2 (referencia) |
| sharedtask, desv. estándar | 0,12 | 0,05 | 0,12 | 0,02 | 0,08 | 0,02 | sharedtask (dev) | sí: paso 2 |
| sharedtask, rango | 0,23 | 0,09 | 0,22 | 0,04 | 0,13 | 0,04 | sharedtask (dev) | sí: paso 2 |
| sharedtask, ensemble de logits | 11,57 | 3,36 | 9,97 | 1,39 | 10,77 | 2,37 | sharedtask (dev) | sí: paso 2 (descartado) |
| sharedtask, ensemble de probabilidades | 10,25 | 3,15 | 9,88 | 1,41 | 10,06 | 2,28 | sharedtask (dev) | sí: paso 2 (descartado) |
| **both s42 (principal)** | **10,90** | **2,35** | **3,30** | **0,38** | **7,10** | **1,37** | sharedtask_both (**heldout**) | **no**: medido una vez, después de decidir |
| both s43 | 10,62 | 2,32 | 3,42 | 0,40 | 7,02 | 1,36 | sharedtask_both (heldout) | no |
| both s44 | 10,83 | 2,32 | 3,42 | 0,39 | 7,13 | 1,36 | sharedtask_both (heldout) | no |
| both, media de 3 semillas | 10,79 | 2,33 | 3,38 | 0,39 | 7,08 | 1,36 | sharedtask_both (heldout) | no |
| both, desv. estándar | 0,14 | 0,02 | 0,07 | 0,01 | 0,05 | 0,01 | sharedtask_both (heldout) | no |
| **sharedtask s42 (entrega)** | **9,71** | **1,76** | **3,62** | **0,43** | **6,66** | **1,09** | sharedtask (**heldout**) | **no** |
| sharedtask s43 | 9,62 | 1,77 | 3,44 | 0,40 | 6,53 | 1,08 | sharedtask (heldout) | no |
| sharedtask s44 | 9,78 | 1,78 | 3,62 | 0,42 | 6,70 | 1,10 | sharedtask (heldout) | no |
| sharedtask, media de 3 semillas | 9,70 | 1,77 | 3,56 | 0,41 | 6,63 | 1,09 | sharedtask (heldout) | no |
| sharedtask, desv. estándar | 0,08 | 0,01 | 0,10 | 0,02 | 0,09 | 0,01 | sharedtask (heldout) | no |

**Anexo A: las mismas mediciones con decodificación greedy (sin LM).** En
scripted coincide siempre con v3, porque v3 ya es greedy en scripted; solo
cambian las columnas de spontaneous y las medias.

| Modelo / configuración | scr WER | scr CER | spo WER | spo CER | media WER | media CER | Split (conjunto) | ¿Influyó? |
|---|---:|---:|---:|---:|---:|---:|---|---|
| both s42 | 4,60 | 0,57 | 9,32 | 1,29 | 6,96 | 0,93 | sharedtask_both (dev) | sí: paso 2, comparación sin LM |
| both s43 | 4,97 | 0,64 | 9,54 | 1,33 | 7,26 | 0,98 | sharedtask_both (dev) | sí: paso 2 |
| both s44 | 4,97 | 0,63 | 9,32 | 1,31 | 7,15 | 0,97 | sharedtask_both (dev) | sí: paso 2 |
| both, ensemble de logits | 6,53 | 0,84 | 11,19 | 1,53 | 8,86 | 1,18 | sharedtask_both (dev) | sí: paso 2 |
| both, ensemble de probabilidades | 4,68 | 0,60 | 9,56 | 1,33 | 7,12 | 0,96 | sharedtask_both (dev) | sí: paso 2 |
| both s42, mask 0,2 | 5,05 | 0,66 | 9,37 | 1,32 | 7,21 | 0,99 | sharedtask_both (dev) | no (confirmación) |
| both s42, mask 0,5 | 5,79 | 0,76 | 9,68 | 1,36 | 7,74 | 1,06 | sharedtask_both (dev) | no (confirmación) |
| sharedtask s42 | 9,23 | 3,01 | 9,52 | 1,33 | 9,37 | 2,17 | sharedtask (dev) | sí: paso 2 |
| sharedtask s43 | 9,46 | 2,99 | 9,73 | 1,33 | 9,60 | 2,16 | sharedtask (dev) | sí: paso 2 |
| sharedtask s44 | 9,28 | 2,92 | 9,59 | 1,32 | 9,43 | 2,12 | sharedtask (dev) | sí: paso 2 |
| sharedtask, ensemble de logits | 11,57 | 3,36 | 11,15 | 1,50 | 11,36 | 2,43 | sharedtask (dev) | sí: paso 2 |
| sharedtask, ensemble de probabilidades | 10,25 | 3,15 | 9,83 | 1,35 | 10,04 | 2,25 | sharedtask (dev) | sí: paso 2 |
| both s42 | 10,90 | 2,35 | 3,30 | 0,38 | 7,10 | 1,37 | sharedtask_both (heldout) | no |
| both s43 | 10,62 | 2,32 | 3,46 | 0,40 | 7,04 | 1,36 | sharedtask_both (heldout) | no |
| both s44 | 10,83 | 2,32 | 3,34 | 0,38 | 7,09 | 1,35 | sharedtask_both (heldout) | no |
| sharedtask s42 | 9,71 | 1,76 | 3,56 | 0,43 | 6,63 | 1,09 | sharedtask (heldout) | no |
| sharedtask s43 | 9,62 | 1,77 | 3,46 | 0,41 | 6,54 | 1,09 | sharedtask (heldout) | no |
| sharedtask s44 | 9,78 | 1,78 | 3,50 | 0,41 | 6,64 | 1,10 | sharedtask (heldout) | no |

**Anexo B: análisis, no sistemas.** El peso de un solo hablante en el dev
scripted de `sharedtask` (modelo de entrega, v3). Solo scripted, porque el
análisis es por hablante scripted.

| Subconjunto | Clips | scr WER | scr CER | ¿Influyó? |
|---|---:|---:|---:|---|
| los 7 hablantes | 1.493 | 9,23 | 3,01 | no (análisis) |
| sin `e990fbdf` | 1.434 | 5,71 | 0,93 | no (análisis) |
| solo `e990fbdf` | 59 | 98,67 | 54,06 | no (análisis) |

---

## 11. Glosario de esta sesión

- **Dispersión entre semillas**: cuánto varía el resultado de una misma receta
  solo por el azar del entrenamiento. Aquí se da como desviación estándar
  muestral y como rango (máximo − mínimo) de tres semillas.
- **Ensemble**: combinar las salidas de varios modelos para que sus errores
  aleatorios se compensen.
- **Fusión frame a frame**: combinar las distribuciones de probabilidad de los
  modelos en cada frame antes de decodificar, en vez de combinar los textos
  finales.
- **Media geométrica / aritmética**: multiplicar y sacar la raíz, frente a sumar
  y dividir. Con probabilidades, la geométrica se comporta como un veto (basta un
  valor muy bajo para hundirla); la aritmética sigue a la mayoría.
- **Heldout**: conjunto reservado que no interviene en ninguna decisión, para
  medir al final sin sesgo de selección.
- **Pre-registro**: dejar escrito, antes de medir, qué se va a medir, cuál es el
  resultado principal y qué no se puede cambiar después.
- **Resultado negativo**: una idea que se midió correctamente y no funcionó. Se
  reporta igual que uno positivo: le ahorra a otros probarla.
- **SpecAugment / mask_time_prob**: tapar tramos aleatorios del audio durante el
  entrenamiento para regularizar; `mask_time_prob` controla qué fracción.
- **Sesgo de selección**: el optimismo que adquiere una medida cuando se ha usado
  para elegir entre opciones. Es por lo que el dev deja de ser imparcial.

---

## 12. Preguntas para profundizar con una IA

**Ensembles y CTC**
- ¿Por qué la media geométrica es la fusión "natural" para modelos de
  clasificación pero falla con CTC? ¿Qué es la "peakiness" de CTC?
- ¿Qué alternativas hay a la fusión frame a frame para modelos CTC (ROVER,
  combinación de hipótesis, alinear los posteriores antes de fusionar)?
- ¿Cómo se mide si los errores de dos modelos están correlacionados, y por qué
  eso limita lo que puede aportar un ensemble?
- ¿Por qué un ensemble de modelos ajustados desde el mismo checkpoint ayuda menos
  que uno de modelos entrenados desde cero?

**Regularización**
- ¿Qué hace exactamente SpecAugment en wav2vec 2.0 y por qué las recetas de
  fairseq usan máscaras mucho mayores que HuggingFace?
- ¿Cuándo ayuda la regularización y cuándo solo quita señal? ¿Cómo se relaciona
  con la duración del entrenamiento?

**Metodología**
- ¿Por qué un conjunto heldout solo sirve si se mira una vez? ¿Qué es el
  "garden of forking paths"?
- ¿Cómo se reporta un resultado con tres semillas: media y desviación, rango,
  intervalos de confianza? ¿Qué se puede afirmar con n = 3?
- ¿Por qué es valioso publicar resultados negativos?
