# Sesión en la máquina GPU (7 de octubre de 2026): qué se hizo y por qué

Este documento explica, con todo el detalle necesario para entenderlo a fondo,
lo que se hizo en la primera sesión de trabajo sobre la máquina con GPU: qué
problemas aparecieron, cómo se resolvió cada uno, qué se midió y, sobre todo,
**por qué** se tomó cada decisión. Está pensado para leerlo de principio a fin
o para dárselo a una IA y hacerle preguntas sobre cualquier sección.

Los números son los medidos en esta sesión. Cuando algo es una hipótesis y no
un hecho comprobado, lo digo explícitamente.

---

## Índice

0. [Resumen en una página](#0-resumen-en-una-página)
1. [Contexto mínimo para entender lo demás](#1-contexto-mínimo-para-entender-lo-demás)
2. [Puesta en marcha del entorno](#2-puesta-en-marcha-del-entorno)
3. [El bug de la métrica durante el entrenamiento](#3-el-bug-de-la-métrica-durante-el-entrenamiento)
4. [El entrenamiento por defecto y el sobreajuste](#4-el-entrenamiento-por-defecto-y-el-sobreajuste)
5. [El modelo de lenguaje y el beam search](#5-el-modelo-de-lenguaje-y-el-beam-search)
6. [El hallazgo clave: el test no reutiliza las frases](#6-el-hallazgo-clave-el-test-no-reutiliza-las-frases)
7. [Ajuste final del LM y primera entrega](#7-ajuste-final-del-lm-y-primera-entrega)
8. [El split "both" y el experimento en curso](#8-el-split-both-y-el-experimento-en-curso)
9. [Spontaneous y las personas mayores: el problema sin resolver](#9-spontaneous-y-las-personas-mayores-el-problema-sin-resolver)
10. [Inventario de cambios (para el commit)](#10-inventario-de-cambios-para-el-commit)
11. [Decisiones abiertas y próximos pasos](#11-decisiones-abiertas-y-próximos-pasos)
12. [Glosario](#12-glosario)
13. [Preguntas para profundizar con una IA](#13-preguntas-para-profundizar-con-una-ia)
14. [Segunda parte (8 de octubre): pasos, LR, memorización y la entrega final](#14-segunda-parte-8-de-octubre-pasos-lr-memorización-y-la-entrega-final)

---

## 0. Resumen en una página

**Lo que se pidió:** (1) verificar el entorno y reproducir el baseline de 13,15
WER; (2) lanzar el entrenamiento por defecto y comprobar si se recuperan los
~8 puntos de ortografía; (3) preparar el LM de subpalabras con beam search.

**Lo que salió**, en la métrica de dev (WER / CER, en %):

| Sistema | Scripted | Spontaneous | Media |
|---|---|---|---|
| Baseline convertido (`ft_cpt_silver`) | 16,08 / 3,77 | 10,23 / 1,39 | 13,15 / 2,58 |
| Reentrenado en texto curado, **paso 500**, greedy | 8,74 / 2,87 | 9,83 / 1,43 | 9,28 / 2,15 |
| Paso 500 + **LM de caracteres + beam search** | **7,94** / 2,93 | **9,11** / 1,37 | **8,52** / 2,15 |

**Los cinco hallazgos que más importan:**

1. **El test scripted NO reutiliza las 2.067 frases de entrenamiento.** Hasta
   ahora se daba por hecho que sí. Cambia cómo se mide todo (sección 6).
2. **El entrenamiento por defecto sobreajusta después del paso 500.** Los puntos
   de ortografía vuelven enseguida; el resto de los 8.000 pasos empeora el modelo
   (sección 4).
3. **La métrica que calculaba el entrenamiento estaba corrompida** (decía 24,65
   para un modelo que en realidad daba 10,95), y elegía el mejor checkpoint
   mirando solo scripted. Ambas cosas arregladas (sección 3).
4. **El LM aporta ~0,8 puntos de WER en cada dominio**, medido de forma honesta.
   Si se hubiera ajustado ingenuamente habría parecido aportar 4 puntos, y en el
   test habría empeorado el resultado (secciones 5–7).
5. **No existe audio de personas mayores de 60 años con transcripción humana**,
   y la mitad spontaneous del test son solo mayores. No podemos medir bien esa
   mitad del ranking (sección 9).

**Entregas generadas (no subidas), todas con el modelo del paso 500:**
`submission/ckpt500_lm_v2/qxp.zip` (**recomendada**: LM con α y β robustos,
sección 8.4), `submission/ckpt500_lm/qxp.zip` (LM, primera configuración) y
`submission/qxp.zip` (greedy).

**Actualización del 8 de octubre (sección 14), que sustituye a lo anterior:**
la entrega recomendada es ahora **`submission/final_n500_lr2e-5_v3/qxp.zip`**
(modelo de 500 pasos con LR 2e-5; scripted sin LM y spontaneous con LM). Entre
250 y 1.000 pasos hay una meseta: todas las recetas empatan dentro del ruido
entre semillas. Haber oído las frases de dev infla el WER scripted del dev
normal en ~1,1 puntos (−21 %).

---

## 1. Contexto mínimo para entender lo demás

### 1.1 Cómo se puntúa la tarea

El concurso mide dos cosas en dos dominios:

- **WER** (*Word Error Rate*): cuántas palabras hay que sustituir, borrar o
  insertar para convertir la transcripción del modelo en la de referencia,
  dividido por el número de palabras de la referencia.
  `WER = (S + D + I) / N`.
- **CER** (*Character Error Rate*): lo mismo, pero contando caracteres.

Los dos dominios son **scripted** (frases leídas en voz alta) y
**spontaneous** (respuestas libres a preguntas). El ranking es una *media de
rankings* sobre cuatro números: WER y CER en cada dominio. Por eso un modelo
que gana en scripted y pierde en spontaneous no gana. Todo el trabajo intenta
mejorar los cuatro a la vez.

Un ejemplo de por qué WER y CER cuentan historias distintas:

```
referencia:  wasinpi kashan
hipótesis:   wasimpi kashan
```

Hay una sola letra mal (`n` → `m`), así que el CER es 1/14 ≈ 7 %. Pero esa letra
hace que la palabra entera cuente como errónea, y el WER es 1/2 = **50 %**. Este
ejemplo no es casual: escribir `n` antes de `p` es una de las convenciones del
texto curado por los organizadores. Por eso la ortografía pesa tanto en el WER.

### 1.2 El modelo: XLS-R + CTC

- **XLS-R 300M** es un modelo de Meta (familia wav2vec 2.0) preentrenado con
  audio de muchas lenguas. Recibe la forma de onda y produce una representación
  cada **20 ms**; a cada uno de esos instantes se le llama **frame**.
- El paper del que parte este repositorio le hizo **CPT** (*continued
  pretraining*: seguir preentrenando con audio quechua sin transcripciones) y
  luego un *fine-tuning* con transcripciones. El checkpoint resultante es
  `QuechuaBase/xls-r-cpt-qxp-silver`, que en este repo se convierte del formato
  fairseq al de HuggingFace (`checkpoints/hf/ft_cpt_silver`).
- Encima va una capa **CTC** (*Connectionist Temporal Classification*). Para
  cada frame da una distribución de probabilidad sobre **50 símbolos**: las
  letras, el apóstrofo, el separador de palabras `|`, unos cuantos símbolos
  heredados de la ortografía antigua (dígitos, vocales con tilde) y un símbolo
  especial llamado **blank** (`<pad>`), que significa "aquí no se emite nada".

**Decodificación greedy** (la que se usaba hasta ahora): en cada frame se coge
el símbolo más probable, se juntan los repetidos consecutivos y se quitan los
blanks.

```
frames:          k k _ a _ _ s s _ a
juntar repetidos k _ a _ s _ a
quitar blanks    kasa
```

Una letra doble necesita un blank entre medias (`l _ l` → `ll`). Greedy decide
cada frame por separado: no sabe qué palabra está formando. Esa es la
debilidad que ataca el LM (sección 5).

### 1.3 Por qué el texto curado vale ~8 puntos

Los organizadores publicaron `validated_sentences_curated.tsv`: las 2.067
frases del corpus escritas con una ortografía unificada del quechua de Puno.
**627 frases cambian** respecto al texto original. El baseline se entrenó con
el texto original, así que escribe la ortografía vieja. Puntuando las mismas
hipótesis contra el texto original da 8,13 % de WER en scripted; contra el
curado, 16,08 %. Esa diferencia (7,95 puntos) no es de acústica, es de
ortografía, y por eso reentrenar con el texto curado es la mejora principal.

### 1.4 Cómo están divididos los datos

- **train**: con lo que se entrena.
- **dev**: con lo que se mide durante el desarrollo y se eligen cosas
  (checkpoint, hiperparámetros).
- **heldout**: reservado para una medida final, sin tocar.

El scripted tiene 22.727 grabaciones de **solo 2.067 frases**: cada frase la
leen ~11 personas. El split por defecto es **disjunto por hablante**: los
hablantes de dev no aparecen en train. Pero, como todos leen las mismas frases,
**las 1.188 frases de dev también están en train**, leídas por otras personas.
Esto será importante en la sección 6.

---

## 2. Puesta en marcha del entorno

La máquina: **RTX 5090** (32 GB), 32 núcleos, 123 GB de RAM, Ubuntu, Python 3.12.
No había venv, ni datos descargados, ni checkpoints; solo el audio de test,
que copiaste a mano.

Cada problema que apareció, con su causa y su arreglo:

### 2.1 La API key estaba en un archivo versionado

**Síntoma:** `git status` mostraba `shared-task/.env.example` modificado, con
tu `MDC_API_KEY` real dentro.

**Por qué es un problema:** `.env.example` está en git (es la plantilla). Con un
commit y un push, la clave habría quedado pública en GitHub para siempre,
incluso borrándola después, porque queda en el historial. El archivo correcto
es `.env`, que está en el `.gitignore`.

**Arreglo:** copiar `.env.example` → `.env` (con permisos 600, solo legible por
ti) y devolver `.env.example` a su versión del repositorio (clave vacía).
Comprobé con `git log -S` que la clave no aparece en ningún commit anterior, así
que no hace falta rotarla.

### 2.2 La GPU necesita una versión concreta de PyTorch

**Causa:** la RTX 5090 es de la arquitectura Blackwell, con *compute capability*
12.0 (`sm_120`). PyTorch trae compilados los kernels de CUDA para una lista de
arquitecturas, y las builds para CUDA 12.4 (las que sugería `requirements.txt`)
no incluyen `sm_120`. Con ellas, PyTorch se instala pero falla al ejecutar en
la GPU.

**Arreglo:** instalar torch desde el índice de **CUDA 12.8** (`cu128`). Lo
verifiqué ejecutando una multiplicación de matrices en la GPU. Actualicé el
comentario de `requirements.txt`.

### 2.3 pip instaló transformers 5, que rompe dos argumentos

`requirements.txt` pide `transformers>=4.45`, y pip instaló la última versión,
la **5.19**. La versión 5 eliminó dos argumentos de `TrainingArguments` que
usaba `03_train.py`:

- **`warmup_ratio`** (qué fracción de los pasos dedicar a subir el *learning
  rate* desde 0). Lo sustituí por `warmup_steps=int(max_steps * warmup_ratio)`:
  un número entero de pasos. Funciona igual en la versión 4 y en la 5. En la 5,
  `warmup_steps` acepta también un decimal entre 0 y 1 como proporción, pero en
  la 4 ese decimal se trataría como "0,1 pasos", es decir, sin warmup. El
  entero es la única forma que funciona en ambas.
- **`group_by_length`**. Estaba puesto a `False`, que es el valor por defecto,
  así que quitarlo no cambia nada (el agrupado por longitud lo hace el sampler
  propio del script).

Para no ir descubriendo argumentos de uno en uno, comprobé todos los que usa el
script contra la firma de la versión 5. Solo fallaban esos dos.

### 2.4 No había `curl`

`run_all.sh` descargaba el checkpoint fairseq (3,6 GB) con `curl`, que no está
instalado en esta máquina. Lo sustituí por `huggingface_hub`, la librería que
transformers ya instala siempre. También reanuda descargas cortadas, y así
desaparece una dependencia del sistema.

### 2.5 Los archivos de Mozilla traen un directorio con la versión

Los archivos se extraen en `data/scripted/cv-corpus-27.0-2026-09-11/qxp/` y
`data/spontaneous/sps-corpus-5.0-2026-09-11-qxp/`. Tanto el constructor de
manifests como `run_all.sh` esperan `data/scripted/qxp/` y
`data/spontaneous/ss-corpus-qxp.tsv`. En el Mac seguramente esas rutas ya
existían de antes. Añadí a `00_download.py` una función que, si tras extraer
hay un único directorio con `corpus-` en el nombre, sube su contenido un nivel.
La probé con estructuras sintéticas de los dos casos y con el caso ya aplanado,
donde no debe hacer nada.

### 2.6 `run_all.sh --only evaluate` no evalúa el baseline

Esa etapa evalúa `$OUT`, que por defecto es el modelo reentrenado
(`ft_curated`). Para el baseline hay que hacer
`OUT=checkpoints/hf/ft_cpt_silver bash shared-task/run_all.sh --only evaluate`.
No lo cambié en el script; lo documenté.

### 2.7 El baseline se reproduce exacto

Resultado: **16,08 / 3,77 scripted, 10,23 / 1,39 spontaneous, media 13,15 /
2,58**, idéntico al Mac, en 24 segundos. Esto valida varias cosas a la vez: la
conversión fairseq→HF, los manifests, la normalización del texto y que la
inferencia funciona bien con transformers 5.

---

## 3. El bug de la métrica durante el entrenamiento

### 3.1 Cómo apareció

Antes de lanzar 2 horas de GPU hice un *smoke test*: 30 pasos de entrenamiento
con una evaluación cada 15, solo para comprobar que todo funciona. La
evaluación interna del entrenamiento dijo esto:

| | Scripted | Spontaneous | Media WER |
|---|---|---|---|
| Baseline | 16,08 / 3,77 | 10,23 / 1,39 | 13,15 |
| Tras 30 pasos, según el entrenamiento | 33,98 / **17,86** | 15,32 / 4,76 | **24,65** |

Que el CER pase de 3,77 a 17,86 en 30 pasos es muy sospechoso. Evalué el mismo
modelo con `04_evaluate.py` y salió **12,34 / 3,59 y 9,56 / 1,31, media 10,95**.
El modelo había **mejorado**; la que fallaba era la métrica.

### 3.2 Causa 1: el relleno (*padding*) de audio

Para aprovechar la GPU se evalúan varios clips a la vez en un **lote**
(*batch*). Los clips duran distinto, así que los cortos se rellenan con ceros
hasta la longitud del más largo. El modelo produce frames también para esa zona
de ceros, y en esos frames el símbolo más probable **no siempre es blank**. Si
se decodifica todo el ancho, aparecen letras basura al final de los clips
cortos.

`04_evaluate.py` ya lo resolvía: con la *attention mask* (la máscara que dice
qué parte de cada clip es audio real) calcula cuántos frames reales tiene cada
clip y recorta antes de decodificar. Su docstring dice que esto había llevado
un CER de 0,99 % a 20,06 % en una prueba. La evaluación de `03_train.py` no
recortaba.

### 3.3 Causa 2: el `-100` del Trainer

El `Trainer` de HuggingFace junta las predicciones de todos los lotes en una
sola matriz. Como cada lote tiene una longitud distinta, rellena con el valor
**-100**. Al decodificar, el tokenizador no conoce el id -100 y lo convierte
en el texto literal `<unk>`. Lo comprobé:

```
ids = [a, a, blank, |, k, -100, -100]   →   'a k<unk>'
```

Ese `<unk>` se pega a la última palabra, así que esa palabra cuenta como error
en casi todos los clips.

### 3.4 Por qué importaba mucho

1. **La elección del mejor checkpoint usa esa métrica.** El entrenamiento guarda
   un checkpoint cada 500 pasos y al final se queda con el mejor
   (`load_best_model_at_end`). Con un número corrupto podía quedarse con el
   checkpoint equivocado.
2. **`dev_metrics.json` y el log son lo que acabaría en el paper.** Alguien
   leyendo el log concluiría que entrenar empeora el modelo (24,65 frente a
   13,15) cuando en realidad lo mejora.

### 3.5 El arreglo

- En `prediction_step` (el paso de evaluación del Trainer) calculo cuántos
  frames reales tiene cada clip y, en los frames de relleno, fuerzo que el
  único símbolo posible sea blank: pongo todas las probabilidades al mínimo
  salvo la del blank.
- En `compute_metrics` convierto los `-100` en blank antes de decodificar.
  Como el blank desaparece al decodificar CTC, el relleno ya no deja rastro.

**Prueba de aceptación:** repetí el smoke test. La métrica interna dio 12,32 /
3,59 y 9,61 / 1,32; `04_evaluate.py` sobre el mismo modelo, 12,32 / 3,59 y
9,59 / 1,31. Ya coinciden: la diferencia de 0,02 en spontaneous viene de que los
dos scripts normalizan el texto de forma ligeramente distinta.

### 3.6 Segundo problema: elegía el checkpoint solo por scripted

El script tenía `metric_for_best_model="eval_scripted_wer"`: el mejor
checkpoint era el de mejor WER **scripted**, ignorando spontaneous. HANDOFF
dice explícitamente que nunca hay que optimizar scripted solo, porque el
ranking pesa los dos dominios por igual.

**Arreglo:** el Trainer ahora calcula `eval_mean_wer` (media del WER de los dos
dominios) y elige por esa métrica. Usé la media del WER y no la de los cuatro
números porque el CER es mucho más pequeño en magnitud y el WER dominaría la
media de todos modos; es una simplificación razonable, no la única posible.

**Por qué importó de verdad:** con el criterio antiguo, el run por defecto se
habría quedado con el paso 1500 (scripted 8,56, el mejor), que tiene
spontaneous en 11,27, **peor que el baseline** (10,23).

---

## 4. El entrenamiento por defecto y el sobreajuste

### 4.1 Qué hace el run por defecto

- **Warm start**: parte del baseline ya entrenado (`ft_cpt_silver`), no de cero.
- Entrena con el **texto curado**.
- **8.000 pasos**, *learning rate* máximo 5e-5 con un 10 % de warmup y bajada
  lineal hasta 0.
- Los lotes se forman por **duración de audio**, no por número de clips:
  160 s de audio por lote (calculado a partir de la memoria de la GPU) × 4 de
  acumulación de gradiente = **640 s de audio por actualización**.
- Train son 49,84 h, así que una pasada completa por los datos (una **época**)
  son ~330 pasos, y **8.000 pasos son unas 24 épocas**.

Tardó ~1 h 45 min a 1,28 pasos/s, con la GPU al 93 %.

### 4.2 Resultados

| Paso | Scripted WER / CER | Spontaneous WER / CER | Media WER |
|---|---|---|---|
| baseline | 16,08 / 3,77 | 10,23 / 1,39 | 13,15 |
| **500** | **8,74 / 2,87** | **9,83 / 1,43** | **9,28** |
| 1000 | 8,91 / 3,01 | 10,93 / 1,52 | 9,92 |
| 1500 | 8,56 / 3,15 | 11,27 / 1,65 | 9,91 |
| 2500 | 10,06 / 4,08 | 10,88 / 1,65 | 10,47 |
| 3000 | 8,67 / 3,47 | 10,67 / 1,57 | 9,67 |
| 8000 | 10,31 / **5,03** | 11,00 / 1,57 | 10,66 |

El modelo final es el del **paso 500** (gracias al criterio de la media), en
`checkpoints/hf/ft_curated`. Hay una copia en `checkpoints/snap/checkpoint-500`.

### 4.3 Cómo leerlo

- **Los primeros 500 pasos traen casi toda la mejora.** Tiene sentido: el modelo
  ya sabía reconocer quechua y solo tenía que reaprender las convenciones
  ortográficas. Eso es poco que aprender y se aprende rápido.
- **A partir de ahí, sobreajuste**: la *loss* de entrenamiento sigue bajando
  (0,30 → 0,15 en el paso 1500) mientras dev empeora. El modelo empieza a
  memorizar los datos de train en lugar de generalizar. El CER scripted acaba
  en 5,03, peor que el baseline.
- **Spontaneous empeora casi desde el principio.** Mi hipótesis principal (no
  comprobada todavía): de las 23 h de spontaneous en train, ~90 % son
  transcripciones **silver**, generadas automáticamente por otro modelo
  (omniASR). Entrenar más es aprender más de sus errores.

### 4.4 ¿Se recuperaron los 8 puntos?

**En scripted, sí: 7,3 de 7,95** (16,08 → 8,74 en el paso 500). Pero la sección
6 matiza el 8,74: está medido sobre frases que el modelo ya había oído, y eso lo
favorece.

### 4.5 Por qué no lo paré

Lo dejé terminar porque era el run que pediste, porque el paso 500 estaba
protegido (es el que se queda como modelo final) y porque la curva completa es
útil para el paper. Lo que enseña: **8.000 pasos con este LR son demasiados
para un warm start**. El próximo entrenamiento debería ser mucho más corto.

---

## 5. El modelo de lenguaje y el beam search

### 5.1 La idea

Greedy decide cada frame por separado. El **beam search** mantiene en paralelo
las K mejores hipótesis parciales (K = *beam*, aquí 64), las va extendiendo
frame a frame y descarta las peores. A cada hipótesis se le suma la opinión de
un **modelo de lenguaje** (LM), que puntúa lo plausible que es el texto en
quechua. La puntuación de una hipótesis es:

```
puntuación = Σ log p_CTC(símbolo | audio)          ← lo que dice el modelo acústico
           + α · Σ log10 P_LM(carácter | anteriores) ← lo que dice el LM
           + β · (número de separadores de palabra)   ← bonificación por palabra
```

- **α** (`lm_weight`): cuánto se fía el decodificador del LM. Con α = 0 el LM
  no influye y el resultado es igual a greedy.
- **β** (`sil_score`): compensa que el LM tiende a preferir textos con menos
  palabras. Un β positivo anima a separar palabras y uno negativo a juntarlas.

Ambos se ajustan probando combinaciones en dev (*grid search*).

**Por qué prometía:** el perfil de error del modelo son sobre todo
**sustituciones** (palabras casi bien, una letra mal), con pocas inserciones u
omisiones. Ese es justo el tipo de error que un LM corrige: entre `wasinpi` y
`wasimpi`, que suenan casi igual, el LM sabe cuál es habitual.

### 5.2 Por qué de caracteres y no de palabras

El quechua es **aglutinante**: las palabras se forman encadenando sufijos.
`llaqtaykupiqa` es `llaqta` (pueblo) + `-y` (mi) + `-ku` (plural exclusivo) +
`-pi` (en) + `-qa` (tópico). Como resultado, **el 77 % de las palabras
distintas del corpus aparece una sola vez**. Un LM de palabras nunca habría
visto la forma correcta de la mayoría de las palabras que tiene que juzgar.

Un **n-grama de caracteres** de orden 6 a 12 (que mira los 5–11 caracteres
anteriores) abarca uno o dos morfemas. Aprende, por ejemplo, que después de
`kuna` es muy probable `pi` o `qa`. Generaliza a palabras nunca vistas porque
lo que aprende son las piezas. Es el "LM de subpalabras" que pedía
IMPROVEMENTS, y además encaja exactamente con lo que emite el modelo
acústico, que también son caracteres.

### 5.3 Por qué flashlight y no pyctcdecode

IMPROVEMENTS sugería `pyctcdecode`. El problema es que pyctcdecode aplica el LM
**solo cuando una palabra termina**, y contra un vocabulario de palabras. Para
un LM de caracteres hace falta que el LM puntúe **cada carácter** según se
emite. El decodificador *lexicon-free* de **flashlight-text** hace exactamente
eso: puntúa el LM en cada símbolo nuevo, incluido el separador `|`. Lo
comprobé leyendo su código fuente (`LexiconFreeDecoder.cpp`).

### 5.4 Cómo se construye el LM

**Qué es un LM de n-gramas.** Estima la probabilidad del siguiente símbolo
dados los n−1 anteriores, contando en un texto de entrenamiento. El problema es
que muchas secuencias no aparecen nunca en el texto, y a esas no se les puede
dar probabilidad cero. Los métodos de **suavizado** resuelven eso.

**Kneser-Ney modificado** (el estándar, el que usa KenLM) combina tres ideas:

1. **Descuento:** a cada secuencia vista se le resta un poco de su cuenta, y esa
   masa de probabilidad se reserva para lo no visto. Se usan tres descuentos
   distintos según si la secuencia se vio 1, 2, o 3+ veces (la parte
   "modificado"), y se calculan a partir de cuántas secuencias se vieron
   exactamente 1, 2, 3 y 4 veces.
2. **Interpolación:** la probabilidad de orden n se mezcla con la de orden n−1
   (contexto más corto), y así recursivamente hasta un reparto uniforme.
3. **Cuentas de continuación:** en los órdenes bajos no se cuenta cuántas veces
   aparece una secuencia, sino **tras cuántos contextos distintos** aparece.
   Una terminación como `qa`, que sigue a cientos de raíces distintas, recibe
   mucha probabilidad de continuación. Una secuencia que solo aparece dentro de
   una palabra concreta recibe poca, aunque esa palabra sea frecuente. Esto
   hace que el orden bajo sea bueno justo para los casos donde se usa: cuando
   el contexto largo nunca se vio.

**Por qué lo programé yo.** La herramienta estándar para entrenar estos LMs es
`lmplz`, de KenLM. Para compilarla hacen falta Boost y CMake, y no había Boost
ni sudo para instalarlo. Escribí el mismo algoritmo en Python (`ctc_lm.py`,
función `build_arpa`) y lo guardo en el formato estándar **ARPA**, que
cualquier herramienta lee: cada línea es una secuencia con su log-probabilidad
y su peso de *backoff*.

**Cómo verifiqué que está bien.** Una distribución de probabilidad tiene que
sumar 1. `check_arpa` carga el LM con KenLM (la implementación de referencia)
y, para 200 contextos, suma la probabilidad de todos los caracteres posibles.
La mitad de los contextos son trozos de frases reales (de cualquier longitud)
y la otra mitad secuencias aleatorias, que fuerzan a usar los caminos de
*backoff*. En los 25+ LMs construidos, el error máximo fue de ~10⁻⁶. Si alguna
vez supera 10⁻³, el script se niega a guardar el LM.

**Una decisión de datos:** cada frase scripted cuenta **una vez**, aunque la lean
11 personas. Esa repetición es un artefacto de cómo se recogió el corpus, no
refleja la frecuencia real de las frases en la lengua.

**Fuentes de texto** (siempre solo de `train.tsv`; meter texto de dev en el LM
falsearía la medida):

| Fuente | Líneas | Caracteres | Qué es |
|---|---|---|---|
| `scripted` | 2.051 | 69.707 | frases scripted distintas |
| `scripted_nodev` | 863 | 29.077 | las mismas, sin ninguna frase que esté en dev |
| `spont_gold` | 588 | 69.129 | transcripciones humanas de spontaneous |
| `spont_all` | 6.015 | 645.677 | lo anterior + transcripciones silver |
| `all` | 8.066 | 715.384 | scripted + spont_all |
| `all_nodev` | 6.878 | 674.754 | all, sin las frases de dev |

### 5.5 El límite de orden 6 y la recompilación

Las versiones de kenlm y flashlight-text que instala pip vienen compiladas con
un **orden máximo de 6**, y queríamos probar hasta 12. Hubo que recompilar las
dos desde el código fuente con orden máximo 16.

**El riesgo oculto (ABI).** flashlight-text, al compilarse, enlaza con la
librería de kenlm. KenLM guarda su estado interno en una estructura (`State`)
cuyo **tamaño depende del orden máximo** con el que se compiló. Si flashlight
se compila pensando en orden 6 y kenlm en orden 16, cada uno interpreta la
memoria del otro con un tamaño distinto. Eso no da error: da probabilidades
basura en silencio. Por eso compilé los dos con el mismo valor (hizo falta un
pequeño parche en el `setup.py` de flashlight, que no exponía la opción, y un
ajuste para que el CMake 4 aceptara una dependencia antigua). La receta exacta
está en `requirements.txt`.

**Cómo lo verifiqué:**

1. **Token a token:** comparé la probabilidad que da el KenLM de flashlight con
   la de kenlm-python para cada carácter de 40 frases, en LMs de orden 6, 8
   y 12. **Diferencia: exactamente 0.**
2. **Hipótesis completas:** comparé la puntuación de LM acumulada que reporta el
   decodificador con la puntuación de kenlm-python sobre el mismo texto. Al
   principio no coincidían (ratio ~0,64), y no lo di por bueno hasta entender
   por qué: la lista de tokens que devuelve flashlight incluye un `|` artificial
   al principio y al final (los estados inicial y final del decodificador), que
   no se puntúan con el LM. Al quitarlos, la diferencia quedó en ~10⁻⁵, que es
   precisión de coma flotante.
3. Comprobé también que la primera hipótesis devuelta es siempre la de mayor
   puntuación (100 de 100 clips).

### 5.6 Las herramientas de ajuste

- **`06_build_lm.py`** construye los LMs de cada fuente y orden, los verifica y
  calcula su **perplejidad** en dev. La perplejidad es, a grandes rasgos,
  entre cuántos caracteres "duda" el LM en cada paso; cuanto más baja, mejor
  predice.
- **`07_tune_lm.py`** pasa el modelo acústico **una sola vez** por dev y guarda
  sus log-probabilidades (las "emisiones": una matriz frames × 50 por clip) en
  `checkpoints/emissions/`. Después decodifica todas las combinaciones de LM ×
  α × β desde esa caché, en paralelo en 24 núcleos. Así se pueden probar cientos
  de combinaciones gastando solo ~45 s de GPU.
- **Dos filas de control** en cada ejecución: *greedy* desde la caché (tiene que
  coincidir con `04_evaluate.py`) y *beam search sin LM* (tiene que coincidir
  con greedy). Las dos coincidieron exactamente. Si no, la caché o el
  decodificador estarían mal y ningún otro número sería fiable.
- **Integración:** `04_evaluate.py` y `05_predict.py` aceptan `--lm_config`, un
  JSON con LM, α, β y beam para cada dominio. Para comprobar la integración
  reproduje con `04_evaluate.py` el resultado de una configuración del ajuste;
  salió idéntico (6,92 / 2,78 y 9,42 / 1,40).

### 5.7 Perplejidad por carácter en dev

| LM | Orden 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| `scripted` sobre dev scripted | 4,33 | 2,74 | 2,14 | 1,93 | **1,87** |
| `scripted_nodev` sobre dev scripted | 4,96 | 4,36 | 4,36 | 4,36 | 4,35 |
| `spont_all` sobre dev spontaneous | 4,21 | 3,26 | **3,15** | 3,17 | 3,17 |
| `spont_gold` sobre dev spontaneous | 4,36 | 3,67 | 3,65 | 3,65 | 3,65 |

Cómo se lee:

- Con el LM `scripted`, la perplejidad cae casi a 1 al subir el orden: el LM
  está prácticamente **seguro** del siguiente carácter. Eso no es saber
  quechua, es **recordar las frases**, porque las frases de dev están en su
  texto de entrenamiento.
- Con `scripted_nodev` (sin esas frases), la mejora se estanca en el orden 6:
  para frases nuevas, más contexto no ayuda.
- En spontaneous, el óptimo es el orden 8, y el texto silver ayuda (3,65 → 3,15).

Esta tabla ya apuntaba al problema de la sección siguiente.

---

## 6. El hallazgo clave: el test no reutiliza las frases

### 6.1 La suposición previa

HANDOFF e IMPROVEMENTS suponían que el test scripted salía de las mismas 2.067
frases. Parecía muy sólido: el banco de frases de Common Voice v27 son
**exactamente esas 2.067, todas grabadas**, y el test son frases leídas y
grabadas en navegador, igual que Common Voice.

### 6.2 La trampa que apareció en el primer ajuste

El primer grid (529 combinaciones, 51 minutos) dio esto para scripted:

| α (LM `all_o12`) | WER en dev normal | WER / CER con un LM que no ha visto las frases de dev |
|---|---|---|
| greedy | 8,72 | 8,72 / 2,87 |
| 0,75 | 6,01 | **7,99** / 3,01 |
| 1,0 | 5,39 | 8,12 / 3,15 |
| **2,0** (lo que elige el grid) | **4,54** | 9,83 / 3,56 |

El grid quería α = 2,0, y además en el borde del rango probado: parecía que
cuanto más peso al LM, mejor. Pero con α alto el LM **reconoce frases enteras
que ya vio** y fuerza la hipótesis hacia ellas. Si las frases del test son
nuevas, esa misma configuración da 9,83 / 3,56, **peor que no usar LM**.

Así que todo dependía de una pregunta: ¿las frases del test están en el banco?

### 6.3 Cómo se comprobó

Con tu permiso, usé **solo nuestras propias predicciones** sobre el audio de
test, sin ninguna etiqueta:

1. Decodifiqué con greedy los 438 clips scripted del test.
2. Para cada hipótesis busqué la frase del banco más parecida y medí su
   **distancia de edición normalizada**: el número de caracteres que hay que
   cambiar, insertar o borrar para pasar de una a otra, dividido por la longitud.
   0 es idéntica y ~1 es nada que ver.
3. Para saber qué significan esos números, los **calibré con dev**: dev contra
   el banco completo (caso "la frase está en el banco") y dev contra el banco
   sin sus propias frases (caso "frase nueva").

| | Mediana de la distancia | % de clips a ≤ 0,05 |
|---|---|---|
| dev, frase en el banco | 0,000 | 91 % |
| dev, frase quitada del banco | 0,516 | 0 % |
| **test scripted** | **0,480** | **3,7 %** |

El test se comporta igual que "frase nueva". Solo 16 de 438 clips quedan cerca
de alguna frase conocida (probablemente frases cortas o hechas con las mismas
plantillas). Además, las hipótesis del test tienen 8–12 palabras, mientras que
las frases del banco tienen 4 de media: es otro conjunto de frases.

**Sobre la legitimidad.** No se usó ninguna etiqueta del test ni se entrenó con
él. Aun así, se usó el audio del test para tomar una decisión de configuración.
Te lo pregunté antes de hacerlo y lo aprobaste. Mi recomendación es
**declararlo en el paper** de forma transparente.

### 6.4 Las tres consecuencias

1. **El α alto queda descartado.** Para scripted, el α se ajusta con LMs que no
   han visto las frases de dev (`_nodev`), que es la situación del test.
2. **El dev scripted es optimista también para el modelo acústico.** El modelo ha
   oído cada frase de dev leída por ~11 hablantes de train. Una red neuronal
   puede aprender a "esperar" frases que conoce, así que el 8,74 del paso 500 se
   midió en condiciones más fáciles que las del test. Probablemente el
   sobreajuste de la sección 4 pese más en el test de lo que muestra dev.
3. **Hace falta un dev disjunto en hablante Y en texto** para elegir el
   checkpoint y la duración del entrenamiento. Ver sección 8.

---

## 7. Ajuste final del LM y primera entrega

### 7.1 El ajuste en la condición correcta

Segundo grid, solo con LMs `_nodev` para scripted, α fino entre 0,5 y 1,0 y
β en {−1, 0, 1} (94 combinaciones, 10 minutos):

| LM para scripted | Mejor α | β | WER | CER |
|---|---|---|---|---|
| greedy | — | — | 8,72 | 2,87 |
| **`scripted_nodev_o6`** | **0,625** | **0** | **7,94** | **2,93** |
| `scripted_nodev_o8` | 0,875 | 0 | 8,00 | 3,03 |
| `all_nodev_o10` | 0,75 | 0 | 8,24 | 3,00 |
| `all_nodev_o8` | 0,5 | 0 | 8,32 | 2,96 |
| `all_nodev_o6` | 0,5 | −1 | 8,29 | 3,01 |

Para spontaneous ganó **`spont_all_o8`, α 0,5, β 1**: 9,11 / 1,37, frente a
9,83 / 1,42 en greedy. Con solo texto gold quedaba en 9,44.

**Un resultado curioso:** añadir texto spontaneous al LM de scripted
(`all_nodev`) **empeora** el WER aunque **baja** la perplejidad (4,27 frente a
4,36). La perplejidad mide lo bien que el LM predice el texto correcto. El WER
depende de otra cosa: si el LM ayuda a elegir bien **entre las confusiones
concretas que comete el modelo acústico**. No tienen por qué ir juntas. No sé
con certeza por qué pasa aquí; una hipótesis es que el texto spontaneous (más
coloquial y con errores silver) empuja hacia formas que no son las del registro
escrito de las frases scripted.

### 7.2 Comprobaciones

- **Beam 128 frente a 64:** la misma configuración dio 7,94 / 2,94 frente a 7,94 /
  2,93. Ampliar la búsqueda no encuentra nada mejor, así que beam 64 basta.
- En los dos dominios gana **α 0,5–0,625**, lejos del borde del grid.

### 7.3 La configuración elegida (`checkpoints/lm/submission_v1.json`)

```json
{
  "scripted":    { "lm": "scripted_o6",  "alpha": 0.625, "beta": 0.0, "beam": 64 },
  "spontaneous": { "lm": "spont_all_o8", "alpha": 0.5,   "beta": 1.0, "beam": 64 }
}
```

**Un detalle importante:** el α de scripted se ajustó con `scripted_nodev_o6`,
pero en la entrega se usa **`scripted_o6`** (con todas las frases, incluidas las
de dev). La razón es que para el test **cualquier** LM está en la condición de
"frase nueva", y las frases de dev son texto quechua válido que aporta
información. No hay fuga, porque las frases de dev no son las del test. El
riesgo pequeño es que un LM con 2,4 veces más texto sea algo más fuerte y su α
óptimo un poco distinto. Se puede afinar con los LMs del split `both`
(sección 8).

### 7.4 El efecto en los cuatro números

| | Scripted WER | Scripted CER | Spontaneous WER | Spontaneous CER |
|---|---|---|---|---|
| greedy | 8,72 | 2,87 | 9,83 | 1,42 |
| beam + LM | **7,94** | 2,93 | **9,11** | **1,37** |
| cambio | −0,78 | +0,06 | −0,72 | −0,05 |

Tres de los cuatro números mejoran y uno empeora muy poco. Media: 9,28 / 2,15 →
**8,52 / 2,15**.

### 7.5 Las entregas

- **`submission/ckpt500_lm/qxp.zip`**: paso 500 + LM con `submission_v1.json`.
  Decodificar los 659 clips tardó 54 s. *Superada por `ckpt500_lm_v2`
  (sección 8.4).*
- **`submission/qxp.zip`**: paso 500 en greedy (la generó `run_all.sh`).

Ambas tienen 659 filas en el orden de la plantilla de los organizadores, y cada
clip se decodifica con el LM de su dominio, según la columna `type` de
`qxp_test_dataset.tsv`. Si a algún clip le faltara el tipo, el script falla en
lugar de usar greedy en silencio.

**Un error mío, ya corregido:** la primera versión la generé con
`--out submission/qxp_ckpt500_lm`, y dentro del zip el archivo se llamó
`qxp_ckpt500_lm.tsv`. Los organizadores esperan **`qxp.tsv`**. La regeneré en
`submission/ckpt500_lm/qxp` y borré la mal nombrada. La regla queda documentada
en el README: **cada candidato va en su propio directorio** y `--out` termina
siempre en `/qxp`.

---

## 8. El split "both" y el experimento en curso

### 8.1 El diseño

Hasta ahora había dos formas de dividir el scripted:

- **Por hablante** (la de siempre): hablantes nuevos, pero frases conocidas.
- **Por frase** (`--split_by sentence`): frases nuevas, pero hablantes conocidos.

El test tiene **las dos cosas nuevas**. El nuevo `--split_by both` elige por
separado un conjunto de hablantes de dev y un conjunto de frases de dev:

- **dev** = hablantes de dev leyendo frases de dev;
- **train** = hablantes de train leyendo frases de train;
- los cruces (hablante de dev con frase de train y al revés) se **descartan**.

### 8.2 El coste

El tamaño de dev es aproximadamente (fracción de hablantes) × (fracción de
frases), así que hay que subir ambas para que salga algo útil. Con 10 % de
hablantes y 15 % de frases:

| | Clips | Horas |
|---|---|---|
| dev scripted | 339 (267 frases, 0 compartidas con train) | 0,51 |
| heldout scripted | 364 | 0,47 |
| train scripted | — | 16,71 (frente a 26,61 en el split normal) |
| descartados | 9.435 | 12,80 |

Es un split **de diagnóstico**: sirve para decidir cuántos pasos entrenar y
qué α usar en condiciones realistas. El modelo final se entrena con todos los
datos.

### 8.3 El experimento: `both_s2000`

`checkpoints/hf/both_s2000`: 2.000 pasos desde el baseline sobre este split,
evaluando cada 250 (~30 min). La pregunta era: **¿en qué paso toca fondo el
WER scripted cuando las frases son nuevas?**

| Paso | Scripted WER / CER | Spontaneous WER / CER | Media WER |
|---|---|---|---|
| **250** | **4,60 / 0,63** | **9,83 / 1,44** | **7,22** |
| 500 | 4,90 / 0,62 | 11,07 / 1,58 | 7,99 |
| 750 | 5,27 / 0,69 | 10,04 / 1,44 | 7,66 |
| 1000 | 5,87 / 0,75 | 10,69 / 1,53 | 8,28 |
| 1500 | 5,27 / 0,67 | 10,00 / 1,43 | 7,63 |
| 2000 | 5,42 / 0,70 | 10,02 / 1,41 | 7,72 |

**Cómo leerlo:**

- **El mejor punto es la primera evaluación (paso 250)**, así que el óptimo
  real puede estar incluso antes. Con frases nuevas, el warm start necesita
  todavía menos entrenamiento que en el split normal. Encaja con la idea de la
  sección 4: lo que hay que aprender es la ortografía, y eso se aprende enseguida.
- **Las cifras absolutas no se pueden comparar con las del dev normal.** Que
  aquí salga 4,60 y allí 8,74 **no** significa que las frases nuevas sean más
  fáciles. Este dev tiene solo **4 hablantes y 1.347 palabras** (cada punto de
  WER son ~13 palabras) y no incluye al hablante `e990fbdf`, cuyo 106 % de WER
  infla el dev normal. La diferencia viene sobre todo de **quién habla**. Lo que
  sí es comparable es la **forma de la curva** dentro del mismo split.
- **Este run todavía no responde cuánto memoriza el modelo acústico.** Para
  eso hace falta un experimento A/B sobre este mismo dev: entrenar (A) como
  ahora y (B) añadiendo los clips cruzados que se descartaron (hablantes de
  train leyendo las frases de dev). Si B sale mucho mejor que A en el mismo
  dev, esa diferencia es lo que "haber oído la frase" infla la medida del dev
  normal.

### 8.4 El LM en la condición limpia, y la elección robusta de α y β

Con este modelo, el modelo acústico no ha oído las frases de dev y los LMs de
`checkpoints/lm_both/` tampoco las contienen. Es la medida más parecida al test:

| | Scripted WER / CER | Spontaneous WER / CER |
|---|---|---|
| greedy | 4,60 / 0,63 | 9,83 / 1,44 |
| mejor LM del grid | **4,16 / 0,57** (`scripted_o8`, α 0,5, β −1) | **9,35 / 1,39** (`spont_all_o8`, α 0,25, β 0) |

El LM sigue ayudando, y en esta condición **también mejora el CER scripted**.
Pero los mejores α y β **cambiaron respecto al modelo del paso 500**:

- En spontaneous, la configuración de `submission_v1` (α 0,5, β 1) ganaba
  0,72 puntos de WER con el modelo del paso 500 y **0,00** con este.
- En scripted, el óptimo bajó de α 0,625 a 0,5.

**Por qué no persigo el óptimo de cada grid.** Con devs de 260 o 339 clips,
diferencias de 0,2–0,3 puntos son unas pocas palabras: es **ruido**. Si cada vez
se elige el máximo exacto del grid, se está ajustando al ruido de ese dev
concreto (sobreajuste de hiperparámetros). La alternativa es elegir la
configuración **robusta**: la que tiene mejor ganancia **en el peor caso** entre
los dos modelos medidos (paso 500 con LM `_nodev`, y `both_s2000`).

Resultado: **α 0,5 y β 0 en los dos dominios**:

| | Ganancia WER (modelo paso 500 / both_s2000) | Ganancia CER |
|---|---|---|
| Scripted | +0,57 / +0,30 | −0,08 / +0,05 |
| Spontaneous | +0,41 / +0,14 | +0,02 / 0,00 |

(La ganancia es greedy menos LM, así que positivo = mejora.)

Es una ganancia modesta pero **consistente**: nunca empeora el WER, y el CER se
mueve en ±0,08. Además, usar el mismo α y β en los dos dominios es más fácil de
justificar en el paper. Una regla clara que sale de todo esto: **α ≥ 1 empeora
siempre** que las frases son nuevas.

Esta configuración está en `checkpoints/lm/submission_v2.json`, y el candidato
correspondiente en **`submission/ckpt500_lm_v2/qxp.zip`**. Cambia 29
transcripciones respecto a v1 y 136 de 659 respecto a greedy. **Es ahora el
candidato recomendado**, en lugar de `ckpt500_lm`.

---

## 9. Spontaneous y las personas mayores: el problema sin resolver

La mitad spontaneous del test son 221 clips, **todos de mayores de 60 años**.
Al mirar quién hay en nuestros datos:

- **dev spontaneous:** 258 de sus 260 clips son de **dos hablantes de veintitantos
  años**. No dice casi nada sobre cómo le irá al modelo con personas mayores.
- **En todo el corpus, mayores de 60:** 1 clip con transcripción gold y 17
  pendientes de validar. Los 528 clips restantes (2,06 h, 9 hablantes) solo
  tienen transcripción **silver**.
- **En general:** el spontaneous de train tiene ~2,2 h de gold y ~23 h de silver.

**Qué implica:**

- **No hay forma limpia de medir la mitad del ranking que más pesa.**
- IMPROVEMENTS (ítem 4) proponía dar más peso a los mayores durante el
  entrenamiento. Eso significaría dar más peso a **transcripciones automáticas**,
  y no podríamos medir si ayuda o perjudica.
- La degradación de spontaneous durante el entrenamiento (sección 4) es
  coherente con que el modelo aprenda errores del silver, pero no está probado.

Esta decisión es tuya. Algunas opciones: aceptarlo y declararlo como
limitación en el paper; usar los 17 clips pendientes como una mini-referencia
muy ruidosa; transcribir a mano algunos clips de mayores (si alguien del equipo
sabe quechua de Puno, aunque sean 50 clips ya servirían de referencia);
preguntar a los organizadores.

---

## 10. Inventario de cambios (para el commit)

**Archivos modificados:**

| Archivo | Cambio |
|---|---|
| `shared-task/00_download.py` | `flatten_release_dir`: aplana el directorio con versión de los archivos de Mozilla. |
| `shared-task/01_build_manifests.py` | `--split_by both` (disjunto en hablante y texto), `--dev_text_frac`, `--heldout_text_frac`. |
| `shared-task/03_train.py` | `warmup_steps` entero; sin `group_by_length`; métrica sin el bug del padding ni del `-100`; mejor checkpoint por la media de WER. |
| `shared-task/04_evaluate.py` | `--lm_config` / `--lm_dir`: evaluación con beam search + LM por dominio. |
| `shared-task/05_predict.py` | `--lm_config` / `--lm_dir` / `--metadata`: predicción con el LM del tipo de cada clip. |
| `shared-task/run_all.sh` | Descarga del checkpoint con `huggingface_hub` en lugar de `curl`. |
| `shared-task/requirements.txt` | Nota de cu128 para Blackwell; kenlm y flashlight-text con la receta de orden 16. |
| `shared-task/HANDOFF.md`, `README.md`, `IMPROVEMENTS.md` | Hallazgos, resultados y uso de las herramientas nuevas. |
| `.gitignore` | **Cambio tuyo**: ignora `data/scripted/` y `data/spontaneous/`. |

**Archivos nuevos. No están en git todavía, así que acuérdate del `git add`:**

| Archivo | Qué es |
|---|---|
| `shared-task/ctc_lm.py` | Generador de ARPA (Kneser-Ney modificado), su verificación, el decodificador beam search y las funciones compartidas. |
| `shared-task/06_build_lm.py` | Construye y verifica los LMs por fuente y orden. |
| `shared-task/07_tune_lm.py` | Caché de emisiones y grid search de LM / α / β. |
| `shared-task/EXPLICACION_SESION_2026-10-07.md` | Este documento. |

**Cambios del 8 de octubre** (sección 14):

| Archivo | Cambio |
|---|---|
| `shared-task/01_build_manifests.py` | `--train_on_dev_sentences` (el control de memorización, condición B). |
| `shared-task/05_predict.py` | No escribe el zip si falla algún archivo y termina con error; el log dice si cada dominio usa LM o no. |
| `shared-task/ctc_lm.py` | `"lm": null` en la configuración (= greedy); atributo `has_lm`. |
| `shared-task/04_evaluate.py` | El log dice "greedy" para los dominios sin LM. |
| `shared-task/experiments/*.sh` | **Nuevos** (ya en git): los comandos exactos de todos los runs del 8 de octubre. |
| `shared-task/HANDOFF.md` | Hallazgos H–K y próximos pasos reescritos. |

**Lo que no va a git** (está en el `.gitignore`), pero conviene saber dónde está:

- `checkpoints/hf/final_n500_lr2e-5/`: **el modelo de la entrega actual**.
- `checkpoints/lm/submission_v3.json`: **la configuración de la entrega actual**.
- `checkpoints/hf/exp/`: los 14 modelos de los experimentos del 8 de octubre
  (cada uno con su `dev_metrics.json`). Ocupan ~5 GB cada uno contando el
  checkpoint con el estado del optimizador; se pueden borrar los
  `checkpoint-*` internos si hace falta espacio.
- `data/manifests/sharedtask_both_devtext/` y `..._devtext_eq/`: los splits B y B_eq.

- `checkpoints/hf/ft_curated/`: modelo final (paso 500). Copia en `checkpoints/snap/checkpoint-500/`.
- `checkpoints/lm/*.arpa`, `checkpoints/lm_both/*.arpa`: los LMs.
- `checkpoints/lm/submission_v2.json` (y `submission_v1.json`): **las
  configuraciones de las entregas**. Recomiendo copiarlas a `shared-task/` para
  que queden versionadas, porque sin ellas la entrega no es reproducible (al
  usarlas desde ahí, pasa `--lm_dir checkpoints/lm`).
- `checkpoints/hf/both_s2000/`: el modelo del split disjunto (paso 250).
- `checkpoints/lm/tune__*.csv`: todas las combinaciones probadas, con su WER y CER.
- `checkpoints/emissions/*.pkl`: la caché de emisiones.
- `logs/`: todos los logs (descargas, entrenamientos, ajustes).
- `submission/`: las entregas.
- `data/manifests/sharedtask_both/`: el split nuevo.

---

## 11. Decisiones abiertas y próximos pasos

1. **Duración del entrenamiento.** El run `both_s2000` sitúa el óptimo en 250
   pasos o menos. Siguiente: barrer entre 100 y 300 pasos con evaluaciones cada
   50 y reentrenar con todos los datos usando `--max_steps` igual al óptimo, para
   que el calendario de LR (warmup y bajada) se ajuste a esa duración. Cada 250
   pasos cuestan unos 4 minutos.
1b. **Medir la memorización** con el experimento A/B de la sección 8.3.
2. **Silver sí o no.** El mismo run corto con `--no_silver`. Ojo: el dev
   spontaneous no puede mostrar el efecto en personas mayores.
3. **Reajustar el LM** sobre el modelo que salga del punto 1, idealmente con los
   LMs de `checkpoints/lm_both/`.
4. **Personas mayores** (sección 9): decidir qué hacer sin poder medirlo.
5. **Acentos sí o no** (IMPROVEMENTS ítem 2).
6. **Entregar pronto.** `submission/ckpt500_lm/qxp.zip` ya es válida.
7. **Para el paper:** declarar la comprobación del test de la sección 6.3, la
   curva de sobreajuste de la sección 4 y la diferencia entre el dev normal y el
   disjunto como hallazgo metodológico. Es un resultado interesante en sí mismo:
   evaluar con frases compartidas sobreestima tanto al modelo acústico como al LM.

---

## 12. Glosario

- **ABI** (*Application Binary Interface*): cómo dos piezas de código compilado
  se ponen de acuerdo sobre el tamaño y la disposición de los datos en memoria.
  Si no coinciden, leen basura sin dar error.
- **α / β**: peso del LM y bonificación por palabra en el beam search (sección 5.1).
- **ARPA**: formato de texto estándar para LMs de n-gramas.
- **Attention mask**: máscara que indica qué parte de cada clip de un lote es
  audio real y cuál es relleno.
- **Backoff**: cuando una secuencia larga no se vio en el texto, el LM pasa a un
  contexto más corto, multiplicando por un peso.
- **Batch / lote**: grupo de clips que se procesan juntos en la GPU.
- **Beam search**: búsqueda que mantiene las K mejores hipótesis parciales en
  lugar de solo la mejor.
- **Blank**: símbolo CTC que significa "nada en este frame".
- **Checkpoint**: copia guardada del modelo en un paso concreto del entrenamiento.
- **CPT** (*continued pretraining*): seguir preentrenando con audio sin
  transcribir de la lengua objetivo.
- **CTC**: forma de entrenar y decodificar un modelo de voz que emite un símbolo
  por frame, sin necesidad de alinear audio y texto.
- **Dev / heldout / train**: ver sección 1.4.
- **Disjunto por hablante / por texto**: que ningún hablante, o ninguna frase,
  aparezca a la vez en train y en dev.
- **Emisiones / log-probabilidades**: la salida del modelo acústico, una matriz
  frames × 50 con la log-probabilidad de cada símbolo en cada frame.
- **Época**: una pasada completa por los datos de entrenamiento.
- **Fine-tuning**: entrenar un modelo ya preentrenado para una tarea concreta.
- **Frame**: cada instante de 20 ms en el que el modelo emite una distribución.
- **Gold / silver / pending**: transcripción humana validada / automática /
  humana sin validar.
- **Greedy**: decodificar cogiendo en cada frame el símbolo más probable.
- **Kneser-Ney modificado**: método estándar de suavizado de LMs de n-gramas
  (sección 5.4).
- **KenLM**: librería de referencia para cargar y consultar LMs de n-gramas.
- **Learning rate (LR)**: tamaño de los pasos con los que se ajustan los pesos.
- **Lexicon-free**: decodificación que no exige que las palabras estén en un
  diccionario.
- **LM** (*language model*): modelo que puntúa lo plausible que es un texto.
- **N-grama**: secuencia de n símbolos; un LM de n-gramas predice cada símbolo a
  partir de los n−1 anteriores.
- **Padding / relleno**: ceros que se añaden a los clips cortos de un lote para
  igualar longitudes.
- **Perplejidad**: medida de lo bien que un LM predice un texto. Más baja es
  mejor; 1 significa certeza total.
- **Sobreajuste** (*overfitting*): el modelo mejora en train y empeora en datos
  nuevos porque memoriza en lugar de generalizar.
- **Smoke test**: prueba mínima y rápida para comprobar que algo funciona antes
  de gastar recursos de verdad.
- **Warm start**: empezar a entrenar desde un modelo ya entrenado.
- **Warmup**: los primeros pasos, en los que el LR sube gradualmente desde 0.
- **WER / CER**: tasa de error por palabras / por caracteres (sección 1.1).

---

## 13. Preguntas para profundizar con una IA

Pásale este documento entero y pregúntale cosas como estas:

**Sobre CTC y la decodificación**
- ¿Por qué CTC necesita un símbolo blank? ¿Qué pasaría sin él con las letras dobles?
- ¿Por qué el modelo emite símbolos sobre el relleno de ceros si la attention
  mask le dice que es relleno? (Pista: la máscara impide que los frames reales
  atiendan al relleno, pero las posiciones de relleno se siguen calculando y
  pasando por la capa CTC, y nada en el entrenamiento les obliga a dar blank.)
- ¿En qué se diferencia el beam search de CTC con "max" del que usa "log-add"?

**Sobre el entrenamiento**
- ¿Por qué un warm start converge en 500 pasos y luego sobreajusta?
- ¿Qué alternativas hay a entrenar menos pasos para evitar el sobreajuste (LR
  más bajo, congelar capas, más SpecAugment, early stopping)?
- ¿Cómo puede un modelo acústico "memorizar frases"? ¿Qué parte de la red lo hace?

**Sobre el LM**
- Explícame Kneser-Ney modificado con un ejemplo numérico pequeño.
- ¿Por qué un LM con menor perplejidad puede dar peor WER?
- ¿Cómo se relaciona α con la escala de las log-probabilidades (log10 frente a
  logaritmo natural)?
- ¿Tendría sentido un LM de BPE o de morfemas (Morfessor) en lugar de caracteres?
  ¿Cómo se combinaría con un modelo acústico de caracteres?

**Sobre la metodología**
- ¿Por qué ajustar hiperparámetros en un dev con frases compartidas da una
  estimación optimista? ¿Cómo se llama ese problema en la literatura?
- ¿Es metodológicamente aceptable mirar las predicciones sobre el test sin
  etiquetas para tomar una decisión? ¿Cómo se debería declarar en el paper?
- ¿Cómo se puede estimar el rendimiento en personas mayores sin datos de
  referencia de personas mayores?

**Sobre los datos**
- ¿Qué riesgos tiene entrenar con transcripciones silver? ¿Cómo se podría filtrar
  el silver por calidad?
- Si solo hay 2 h de audio de personas mayores (todo silver), ¿qué técnicas de
  adaptación existen (perturbación de velocidad, aumento de datos, adaptación
  al hablante)?

---

## 14. Segunda parte (8 de octubre): pasos, LR, memorización y la entrega final

Esta sección cuenta lo que se hizo después de escribir el resto del documento.
**Algunas conclusiones de las secciones 7 y 8 cambian aquí**; cuando una cosa
contradice a otra, vale lo de esta sección.

### 14.1 Qué había que decidir

1. **Cuántos pasos entrenar y con qué learning rate**, medido en el dev
   disjunto en hablante y texto (el que se parece al test).
2. **Cuánto infla el dev normal** el hecho de que el modelo haya oído las
   frases de dev en boca de otros hablantes (sección 8.3).
3. **Con qué configuración del LM** decodificar el modelo final.

### 14.2 Cómo se diseñaron los runs, y por qué así

Cada configuración es **un entrenamiento independiente que termina su propio
calendario de LR**. La alternativa barata, entrenar una vez 2.000 pasos y
evaluar los checkpoints intermedios (lo que hizo `both_s2000`), tiene un
defecto: el checkpoint del paso 250 de un run de 2.000 está con el LR casi al
máximo, a medio camino. Un run de 250 pasos, en cambio, hace el warmup, baja el
LR hasta 0 y termina "asentado". Son modelos distintos, y lo que se quiere
saber es cuántos pasos debe tener **el run final**.

Todos parten del baseline (`ft_cpt_silver`). Los comandos están en
`shared-task/experiments/2026-10-08_steps_and_memorisation.sh` y
`2026-10-08_final_and_lr.sh`, para que los números del paper se puedan
reproducir. Los modelos están en `checkpoints/hf/exp/`.

### 14.3 Pasos y LR: una meseta amplia (y una conclusión que tuve que corregir)

| Run | Scripted WER / CER | Spontaneous WER / CER | Media WER / CER |
|---|---|---|---|
| 100 pasos | 5,79 / 0,76 | 9,68 / 1,37 | 7,74 / 1,06 |
| 150 pasos | 5,86 / 0,76 | 9,90 / 1,38 | 7,88 / 1,07 |
| 200 pasos | 5,64 / 0,73 | 9,47 / 1,35 | 7,56 / 1,04 |
| 250 pasos, semilla 42 | 5,20 / 0,65 | 9,28 / 1,32 | 7,24 / 0,99 |
| 250 pasos, semilla 43 | 4,97 / 0,62 | 9,47 / 1,31 | 7,22 / 0,97 |
| 300 pasos | 5,12 / 0,67 | 9,47 / 1,38 | 7,30 / 1,03 |
| 500 pasos | 4,97 / 0,66 | 9,73 / 1,39 | 7,35 / 1,03 |
| 500 pasos, LR 2e-5, semilla 42 | 4,60 / 0,57 | 9,32 / 1,29 | 6,96 / 0,93 |
| 500 pasos, LR 2e-5, semilla 43 | 4,97 / 0,64 | 9,54 / 1,33 | 7,26 / 0,98 |
| 1000 pasos, LR 2e-5 | 4,83 / 0,60 | 9,64 / 1,33 | 7,23 / 0,96 |
| 1000 pasos, LR 1e-5 | 4,90 / 0,64 | 9,42 / 1,31 | 7,16 / 0,98 |

(Sin indicar LR, es 5e-5, el valor por defecto.)

**Qué es una "semilla" y por qué importa.** La semilla fija el azar del
entrenamiento: el orden en que se presentan los clips y qué trozos del audio se
enmascaran (`mask_time_prob`, una forma de aumento de datos). Con otra semilla,
la misma receta da un modelo algo distinto. La diferencia entre dos semillas de
la misma receta es **el ruido**: cualquier diferencia entre recetas menor que
eso no se puede distinguir del azar.

**La corrección.** Con la primera semilla, 500 pasos con LR 2e-5 parecían los
claros ganadores (6,96, el mejor en los cuatro números), y lo dije así. La
segunda semilla dio 7,26. **Fue en buena parte suerte.** Mirando la tabla
entera, todo lo que está entre 250 y 1.000 pasos, con LR de 1e-5 a 5e-5, queda
entre 7,1 y 7,3 de media, dentro de ~0,3 de ruido entre semillas.

**La conclusión sólida** es que hay una **meseta amplia**: por debajo de ~200
pasos el modelo aún no ha aprendido del todo la ortografía, y por encima de
~1.000 empieza a sobreajustar (sección 4). Dentro de la meseta da igual. El
modelo final (500 pasos, LR 2e-5) está dentro, así que no hace falta
reentrenarlo. Esto ya es una lección para el paper: un único run con una
semilla puede llevar a conclusiones falsas cuando el dev es pequeño.

### 14.4 La memorización, medida

**Diseño del experimento** (sección 8.3), siempre a 250 pasos y con el mismo dev:

- **A**: el split disjunto normal; las frases de dev nunca están en train.
- **B**: lo mismo, pero añadiendo a train las lecturas de las frases de dev por
  **hablantes de train** (+3,65 h). Se construye con
  `01_build_manifests.py --split_by both --train_on_dev_sentences`.
- **B_eq**: B, pero quitando 3,64 h de clips de A elegidos al azar, para que
  tenga **exactamente las mismas horas que A** (39,94 h).

**Por qué hace falta B_eq.** Si B saliera mejor que A, no sabríamos si es por
haber oído las frases o simplemente por tener más audio. B_eq iguala la
cantidad de datos, así que la única diferencia con A es haber oído las frases.

**Un detalle que apareció al construirlo.** El split marca el 15 % de las frases
como "de dev", pero en el dev solo entran las que leyeron los 4 hablantes de
dev (267 frases). El resto de frases marcadas se descartaban de A sin
necesidad, y B las recupera (398 clips). No afectan a la medida, porque no
están en el dev, pero son datos extra: otra razón más para usar B_eq.

**Resultado** (dos semillas en A y B_eq):

| Train | WER scripted | CER scripted |
|---|---|---|
| A (frases de dev nunca oídas) | 5,20 · 4,97 | 0,65 · 0,62 |
| **B_eq** (oídas, mismas horas) | **3,79 · 4,23** | **0,48 · 0,55** |
| B (oídas, +3,65 h; una semilla) | 4,68 | 0,60 |

Los rangos de A y B_eq **no se solapan** (el peor B_eq, 4,23, es mejor que el
mejor A, 4,97). **Haber oído las frases de dev en boca de otros hablantes baja
el WER scripted ~1,1 puntos (−21 % relativo) y el CER ~19 %.** Ese es el sesgo
optimista del dev normal, con número. Y probablemente crece cuanto más se
entrena, porque memorizar es justo lo que hace el modelo al sobreajustar.

Dos cosas que no sé explicar del todo y declaro como tales:
- B (una semilla) sale peor que B_eq teniendo más datos. Con una sola semilla y
  un ruido de ~0,4, puede ser azar.
- En B y B_eq el WER **spontaneous** sale ~0,4 peor que en A (9,80–9,88 frente a
  9,28–9,47), aunque los datos spontaneous son los mismos. Puede ser ruido o un
  efecto indirecto de cambiar la mezcla de scripted; no está comprobado.

### 14.5 El LM con la receta final: solo ayuda en spontaneous

Reajusté el LM sobre el modelo de 500 pasos con LR 2e-5 del split disjunto
(LMs de `checkpoints/lm_both/`, que no contienen las frases de dev). Junto con
los dos modelos anteriores, ya son tres mediciones. Ganancia frente a greedy
(positivo = mejora):

| Configuración | Spontaneous WER (3 modelos) | Peor caso |
|---|---|---|
| **α 0,5, β 0** | +0,41 · +0,14 · +0,31 | **+0,14** |
| α 0,25, β 0 | +0,07 · +0,48 · +0,38 | +0,07 |
| α 0,5, β 1 (la de v1) | +0,72 · 0,00 · +0,14 | 0,00 |

En **spontaneous**, α 0,5 / β 0 mejora en los tres modelos sin tocar el CER: es
una ganancia pequeña pero fiable.

En **scripted** la historia es otra. El mejor peor caso de todo el grid es
+0,07 de WER, y con el modelo de la receta final todas las configuraciones
quedan en ±0,15 (unas 2 palabras). **El LM ya no aporta nada medible.** La
explicación más probable es que cuanto mejor es el modelo acústico, menos
errores quedan que un LM entrenado con ~2.000 frases cortas sepa corregir.
Además, las frases del test son más largas y de otra fuente (sección 6), así que
el LM de scripted podría incluso empujar en la dirección equivocada sin que
podamos medirlo.

**Decisión (`submission_v3.json`): scripted sin LM, spontaneous con LM**
(`spont_all_o8`, α 0,5, β 0). Para poder expresarlo, la configuración acepta
ahora `"lm": null`. Verifiqué que eso da exactamente lo mismo que greedy (1.493
de 1.493 clips idénticos).

### 14.6 Un incidente: una entrega rota que parecía válida

Lancé la predicción del test mientras un run de refinamiento ocupaba 27 GB de
la GPU. La inferencia se quedó **sin memoria (OOM)** en 560 de los 659 clips.
`05_predict.py` captura los errores por lote, escribe una fila **vacía** para
cada clip fallido y sigue, así que **generó un zip con 560 filas vacías**, con
solo un aviso al final. Si se hubiera subido sin mirar, ese envío habría tenido
casi 100 % de error.

**El arreglo:** si falla cualquier archivo, el script ya **no escribe el zip**,
lista los fallos y termina con código de error (deja el TSV para poder
inspeccionarlo). Lo probé reproduciendo el OOM a propósito con la GPU ocupada:
falló con código 1 y no escribió el zip. Borré la entrega rota y la regeneré con
la GPU libre.

**La lección práctica:** en esta tarjeta, no ejecutar inferencia (ni
`07_tune_lm.py`) al lado de un entrenamiento.

### 14.7 La entrega actual

**`submission/final_n500_lr2e-5_v3/qxp.zip`**: modelo `final_n500_lr2e-5`
(split completo, 500 pasos, LR 2e-5) con la decodificación v3. Tiene 659 filas,
ninguna vacía, y `qxp.tsv` dentro del zip. Cambia 162 transcripciones respecto
a `ckpt500_lm_v2`.

En el dev normal (recuerda: optimista en scripted):

| | Scripted WER / CER | Spontaneous WER / CER | Media |
|---|---|---|---|
| Paso 500 del run por defecto, greedy | 8,74 / 2,87 | 9,83 / 1,43 | 9,28 / 2,15 |
| **Modelo final + v3** | 9,23 / 3,01 | **9,06 / 1,30** | 9,14 / 2,16 |

Que scripted salga *peor* que el paso 500 aquí no es una mala señal. El paso
500 del run largo entrenó con más LR y memorizó más las frases compartidas, que
es justo lo que este dev premia (14.4). En el dev honesto, las recetas de la
meseta son equivalentes y el run largo es claramente peor. Spontaneous, cuyo
texto de dev nunca está en train, mejora.

### 14.8 Lo más prometedor ahora: promediar modelos

El ruido entre semillas (~0,3) es también una oportunidad. Si se entrenan 3–5
modelos con la misma receta y distinta semilla y se **promedian sus
log-probabilidades** antes de decodificar (*ensembling*), los errores
aleatorios de cada uno tienden a compensarse. Es la ganancia barata que queda:
cada semilla son ~7 minutos de GPU y la inferencia sigue siendo rápida. Hay que
medirlo primero en el split disjunto.
