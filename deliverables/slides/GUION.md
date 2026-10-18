# Guion de la presentación

**SIMBig 2026 · Tarea 1 · Equipo Innova**
Vídeo obligatorio de 15 minutos · sin cámara, solo voz sobre las diapositivas
Mazo: `simbig2026_task1.pptx` (13 láminas)

> Este guion manda sobre las notas del orador incrustadas en el `.pptx`: aquellas son una versión
> condensada, útil en modo presentador, pero el texto cronometrado es el de aquí.

---

## Cómo usar este guion

- El texto en párrafos normales es **lo que se dice**, escrito para leerse en voz alta.
- Las líneas en *cursiva entre corchetes* son **indicaciones**: no se leen.
- Cada lámina lleva su ventana de tiempo. Son acumulativas: al terminar la lámina 7 deberías ir por 7:00.
- Duración prevista: **13:40**, a **135 palabras por minuto** y contando dos segundos de pausa en
  cada cambio de lámina. El límite es 15:00.
- El ritmo manda más que el texto: a 145 palabras por minuto —un ensayo ya cómodo— esto sale sobre un
  minuto más corto; a 125, sobre un minuto más largo. Por eso importa el punto de control de la lámina 7:
  si llegas ahí con retraso, aplica los recortes de la sección «Si vas justo de tiempo».

**Grabación.** Sin cámara basta con capturar la pantalla en modo presentador y narrar. Hazlo de una pasada:
si te trabas, repite la frase entera en vez de parar, que así es trivial de cortar después.

**Números.** Se leen a la española: `9,14` es «nueve coma catorce», `465.000` es «cuatrocientos sesenta
y cinco mil». Di «WER» y «CER» siempre igual que la primera vez, en la lámina 2.

**Nombres de dataset.** `scripted` y `spontaneous` se dicen en inglés, tal como están en pantalla, para
que lo que se oye coincida con lo que se lee.

---

## Reparto del tiempo

| # | Lámina | Ventana | Dura |
|---|---|---|---|
| 1 | Portada | 0:00 – 0:45 | 45 s |
| 2 | El quechua de Puno y por qué ASR | 0:45 – 1:45 | 60 s |
| 3 | Hallazgo 1: la ortografía | 1:45 – 2:45 | 60 s |
| 4 | El sistema | 2:45 – 3:45 | 60 s |
| 5 | Resultado principal | 3:45 – 4:30 | 45 s |
| 6 | Hallazgo 2: el test no reutiliza frases | 4:30 – 5:40 | 70 s |
| 7 | Hallazgo 3: cuánto vale haber oído la frase | 5:40 – 7:00 | 80 s |
| 8 | Cuántos pasos, y cuánto es ruido | 7:00 – 8:10 | 70 s |
| 9 | No funcionó: promediar modelos | 8:10 – 9:15 | 65 s |
| 10 | No funcionó: máscara y modelo de lenguaje | 9:15 – 10:25 | 70 s |
| 11 | El conjunto reservado | 10:25 – 11:30 | 65 s |
| 12 | Limitaciones | 11:30 – 12:35 | 65 s |
| 13 | Lo que nos llevamos | 12:35 – 13:40 | 65 s |

---

# Lámina 1 · Portada

**0:00 – 0:45**

*[En pantalla: título, autores y las tres cifras — 13,15 → 9,14 · 8 puntos · 1,1 puntos]*

Buenos días. Soy Raymond Negron, del Departamento de Computación de la Universidad Católica San Pablo,
en Arequipa. El día de hoy voy a presentar el sistema del equipo Innova para la Tarea 1 del shared task de SIMBig 2026. Es un trabajo conjunto con Rosa
Paccotacya-Yanque.

Comenzamos, tres cifras estructuran la charla. Mejoramos el baseline publicado de **13,15** a
**9,14** de WER medio. La ortografía de la referencia curada vale, ella sola, casi **ocho** puntos de
WER en `scripted`. Y medimos cuánto vale haber oído antes una frase en boca de otros hablantes:
**1,1 puntos**.

---

# Lámina 2 · El quechua de Puno y por qué ASR

**0:45 – 1:45**

*[En pantalla: cuatro tarjetas — 465.000 · 66 h · 4 · 659 — y la tabla de las dos configuraciones del paper]*

El quechua de Puno lo hablan unas **465.000** personas, pero casi nadie lo escribe, porque la
escolarización es en español. Por eso el reconocimiento de voz aquí no es una comodidad: una interfaz
de texto deja fuera a las personas a la que queremos ayudar.

Dos cosas sobre la tarea. Partimos del corpus de **66 horas** que publicaron los organizadores. Y el ranking promedia **cuatro** números: WER y CER —tasa de error de palabra
y de carácter— en `scripted` y en `spontaneous`.

Si miran la tabla de abajo: la configuración del paper con mejor
`scripted` tiene **1,19** de WER, pero **13,6** en `spontaneous`. Entonces, promediar los cuatro números descarta esa configuracion
y premia a la configuración equilibrada..

*[Avanza al decir «qué optimizamos».]*

---

# Lámina 3 · Hallazgo 1: la ortografía vale casi ocho puntos

**1:45 – 2:45**

*[En pantalla: tres ejemplos de grafía, la tabla de dos referencias, y la conclusión en terracota]*

Comenzamos con el primer hallazgo.

El quechua de Puno no tiene ortografía estandarizada, así que los voluntarios de Common Voice escribieron
como supieron. Los organizadores publicaron una referencia curada de las **2.067** frases, y de ellas
**629** cambian. Son cambios de una sola letra —*allqu* pasa a *alqu*, *iskay* a *ishkay*—
pero en WER cada uno cuesta una palabra entera.

Este experimento aísla el efecto. Tomamos el modelo del paper, le hicimos transcribir el conjunto de
dev **una** vez, y puntuamos esas transcripciones contra las dos referencias.
Contra el texto crudo da **8,13**. Contra el curado, **16,08**.

El modelo no cambió entre las dos medidas: es el mismo conjunto de hipótesis. Esos **7,95** puntos de
diferencia son ortografía, no acústica.

*[Pausa después de «no acústica». Es la frase que quieres que recuerden. Avanza.]*

---

# Lámina 4 · El sistema

**2:45 – 3:45**

*[En pantalla: el diagrama del pipeline, a toda página]*

Este es el sistema. Partimos del checkpoint publicado de XLS-R-300M con pre-entrenamiento continuo y
afinamos **500** pasos. Los dos corpus y la referencia curada entran en la normalización, que lo
compacta todo a un alfabeto de **29** símbolos.

De ahí salen **dos** splits, y esta es la parte importante. El de la izquierda separa por hablante, igual
que el baseline, pero comparte las frases entre entrenamiento y desarrollo. El de la derecha separa por
hablante **y** por texto.

El modelo que entregamos se entrena con el primero. Y el segundo nos permite una medición limpia.

Abajo a la izquierda, la decodificación: `scripted` con argmax voraz, `spontaneous` con beam search y un
modelo de lenguaje de caracteres de orden ocho.

*[Recorre el diagrama con el cursor en el mismo orden en que lo narras. Avanza.]*

---

# Lámina 5 · Resultado principal

**3:45 – 4:30**

*[En pantalla: la tabla de siete columnas y las tres conclusiones de abajo]*

Este es el resultado. Mejoramos el baseline de **13,15** a **9,14** de WER medio, sobre el mismo split y
contra la misma referencia curada. Es una comparación simétrica: mismo modelo base y
misma regla de medir.

Y siendo honesto sobre de dónde viene la mejora. En `scripted` ganamos **6,85** puntos, que es del
mismo orden que los **7,95** del desajuste ortográfico. La mayor parte de lo que
ganamos es ortografía, no capacidad acústica en realidad.

Lo que sí conseguimos es mejorar en los **cuatro** números a la vez, como se ve en la tabla, que es lo que exige el ranking.

*[Avanza.]*

---

# Lámina 6 · Hallazgo 2: el test no reutiliza las frases del corpus

**4:30 – 5:40**

*[En pantalla: la tabla de tres condiciones a la izquierda, la tarjeta del 0,480 a la derecha]*

Segundo hallazgo. El corpus tiene solo **2.067** frases y todas están grabadas, así que pensamos
que el test oficial las reutilizaría. Y bueno, lo análizamos.

Transcribimos su audio con nuestro
propio sistema y medimos, para cada hipótesis, la distancia de edición a la frase más parecida del corpus. Calibramos esa distancia con el conjunto de desarrollo, donde
sí sabemos la respuesta.

Una hipótesis de desarrollo cuya frase está en el corpus queda a distancia **cero**. Si la quitamos,
sube a **0,516**. El test da **0,480**: se comporta como el caso de frase no vista.
Por tanto, sus frases son nuevas, y además más largas.

Esto decide cómo se mide todo lo demás: un conjunto de desarrollo cuyas frases el modelo ya ha oído a
otros once hablantes premia a quien recuerda texto, no a quien reconoce habla.

Entonces con esa premisa vayamos al siguiente hallazgo.

*[Avanza: esa última frase abre la lámina siguiente.]*

---

# Lámina 7 · Hallazgo 3: cuánto vale haber oído la frase

**5:40 – 7:00**

*[En pantalla: la tabla A / B_eq / B, y las dos tarjetas de abajo]*

Tercer hallazgo.

El split establecido comparte **todas** sus frases de evaluación con el entrenamiento. Con **2.067** frases y **77** hablantes, no hay otra forma de usar las **22.727** grabaciones.
Es una propiedad del benchmark.

Entonces, diseñamos el experimento de la tabla. El conjunto de desarrollo y la receta son los
mismos en los tres casos; lo único que cambia es qué vio el modelo. **A** no oye las frases de
desarrollo. **B** sí las oye, con **3,65** horas más de audio. Y **B eq** las oye con las
mismas horas que A, quitando clips al azar, para que la única diferencia sea la exposición de las frases.

El peor B-eq sigue siendo mejor que el mejor A.
Haber oído la frase en boca de otros vale **1,1 puntos** de WER, un **21 %** relativo. Y es una cota
inferior, porque la memorización crece con el entrenamiento, y el baseline entrena **8.000** pasos frente
a nuestros 500.

*[Avanza.]*

---

# Lámina 8 · Cuántos pasos entrenar, y cuánto es ruido

**7:00 – 8:10**

*[En pantalla: tres tarjetas — 7,02 · 9,23 · 0,3 — y el párrafo de la corrección]*

Sobre cuánto entrenar. Partiendo de un checkpoint ya adaptado al quechua hacen falta muchos menos pasos
que los **8.000** de la receta original. Entre **250** y **1.000** pasos, con tasas de **1e-5** a
**5e-5**, todo cae entre **7,1** y **7,3**: es una superficie ancha, no un óptimo.

Y aquí una corrección que vale la pena. Una corrida de 500 pasos a 2e-5 dio **6,96** de WER
medio, la mejor en los cuatro números, y la anotamos como ganadora. Pero, una segunda semilla de la receta
**idéntica** dio **7,26**. No fue una medición exacta.

Por eso medimos el ruido con tres semillas y fijamos un umbral de **0,3**: por debajo, una diferencia no
se distingue del azar. Y fíjense en el contraste entre los dos conjuntos de desarrollo: el pequeño tiene
más del doble de dispersión. No solo mide con menos precisión: fabrica diferencias que no existen.

*[Avanza.]*

---

# Lámina 9 · Lo que no funcionó: promediar modelos

**8:10 – 9:15**

*[En pantalla: tabla del ensemble a la izquierda, el mecanismo explicado a la derecha]*

Tres cosas que esperábamos que ayudaran y no lo hicieron. La primera: promediar modelos entre
semillas.

Un ruido de **0,3** entre semillas es justo lo que un ensemble debería eliminar. Pero, no ganamos con ninguna
de las dos formas de combinar.

Y el fallo tiene un mecanismo que pudimos medir. CTC no emite letras de forma continua: emite picos
agudos en frames concretos. Dos modelos entrenados por separado colocan esos
picos en frames contiguos, pero no idénticos. Entonces, promediar logits equivale a una media geométrica, así que
un pico seguro promediado contra un valor casi nulo se pierde y la letra desaparece: los carácteres borrados suben de **21** a **50**.

Promediando probabilidades, que es media aritmética, ya no se pierden letras, pero tampoco se gana nada.
Entonces, entendimos que con las alineaciones de CTC, la regla de fusión importa más que el ensemble.

*[Avanza.]*

---

# Lámina 10 · Lo que no funcionó: máscara y modelo de lenguaje

**9:15 – 10:25**

*[En pantalla: dos mitades — SpecAugment a la izquierda, modelo de lenguaje a la derecha]*

Las otras dos cosas que no funcionaron.

A la izquierda, más regularización. El sobreajuste pasados los mil pasos era el fallo que observábamos, y
SpecAugment lo ataca directamente. Pues subir el enmascarado empeora de forma monótona —**6,81**,
**7,05**, **7,68**— y empeora en los cuatro números. Y, con 500 pasos hay poco tiempo para sobreajustar.

A la derecha, el modelo de lenguaje. En `scripted` ninguna configuración gana de forma medible, así que
el sistema final decodifica `scripted` sin él. En `spontaneous` sí gana sobre el conjunto de desarrollo,
entre **0,12** y **0,31** según la semilla. Pero en el conjunto reservado gana **cero**, y pierde
**0,04** y **0,08**.

Una ganancia ajustada sobre el desarrollo no sobrevivió a un conjunto limpio. Pero, lo mantenemos porque el
efecto está dentro del ruido, pero ya no afirmamos que ayude.

*[Avanza.]*

---

# Lámina 11 · El conjunto reservado, medido una sola vez

**10:25 – 11:30**

*[En pantalla: tabla desarrollo frente a reservado, y las tres conclusiones]*

Esta diapositiva es la que más me importa metodológicamente.

Un conjunto de desarrollo que ha guiado todas las decisiones deja de ser una estimación imparcial. El
nuestro definió los pasos, las tasas, **72** configuraciones de modelo de lenguaje y qué decodificador usar
en cada dominio.

Por eso apartamos un conjunto el primer día, no lo miramos nunca, y lo medimos **una** sola vez, después
de congelar todas las decisiones.

Las medias coinciden: **7,02** contra **7,08**. Pero esa coincidencia esconde dos movimientos grandes y
opuestos: `scripted` dobla su error y `spontaneous` lo divide por tres. Pero, pensamos que es más por composición, no
generalización, ya que el reservado contiene un hablante que él solo da **98,67** de WER sobre **59** clips en `scripted`.

*[Avanza.]*

---

# Lámina 12 · Limitaciones

**11:30 – 12:35**

*[En pantalla: tres tarjetas — 1 · 2 de 9 · 3 — y el bloque en terracota de abajo]*

Identificamos cuatro limitaciones.

La primera, el sistema que entregamos no tiene una medición limpia
propia. Se entrena sobre el split establecido, cuyo entrenamiento contiene las frases de desarrollo. Lo
entregamos por comparabilidad con el baseline y porque ve un **15 %** más de frases distintas.

Segunda: el barrido fue estrecho, **2** hiperparámetros de **9**. Tercera: **3** semillas bastan para desenmascarar victorias aparentes, pero no para separar recetas parecidas.

Y la cuarta, la mitad del ranking no la podemos medir con fidelidad. El test de
`spontaneous` son íntegramente hablantes de sesenta años o más, y en todo el corpus esos hablantes
aportan **una** sola transcripción humana. El dominio que decide la mitad de nuestro puesto es el que
menos podemos evaluar.

*[Avanza.]*

---

# Lámina 13 · Lo que nos llevamos

**12:35 – 13:40**

*[En pantalla: tres tarjetas de cierre, las direcciones a futuro y el enlace al repositorio]*

Y, finalmente, para cerrar, tres cosas.

Mejoramos el baseline publicado de **13,15** a **9,14** de WER medio, sobre el mismo split y con la misma
referencia, y somos explícitos en que casi todo ese avance es ortografía, no acústica.

Segundo, pusimos número al solapamiento de frases del benchmark
establecido. Vale unos **1,1 puntos**, y mostramos que el test del concurso no comparte esas frases.

Y tercero, **tres** resultados negativos reproducibles, con el mecanismo explicado en el caso del
ensemble.

Dos direcciones a futuro. Un conjunto de desarrollo disjunto en texto, construido sobre el split oficial
de Common Voice, daría **679** clips honestos en vez de **339**. Y, sobre todo, transcripciones humanas
de hablantes mayores: harían medible la mitad de este benchmark.

Todo el código y los splits están en el repositorio. Muchas gracias.

*[Deja la lámina en pantalla dos o tres segundos antes de cortar la grabación.]*

---

## Si vas justo de tiempo

Recorta en este orden. Cada corte está escrito para que el párrafo siguiente siga encajando:

1. **Lámina 8**, la anécdota de la corrección (6,96 contra 7,26) — unos **20 s**. Queda el umbral de 0,3,
   que es lo que necesitas para las láminas 9 y 10.
2. **Lámina 10**, la mitad izquierda sobre SpecAugment — unos **25 s**. Di solo «subir el enmascarado
   empeora de forma monótona» y pasa al modelo de lenguaje.
3. **Lámina 12**, las limitaciones segunda y tercera — unos **15 s**. No sacrifiques ni la primera ni la
   cuarta: son las dos que de verdad acotan el resultado.

## Si te sobra tiempo

No alargues el cierre. Añade en la **lámina 6** por qué la calibración usa el conjunto de desarrollo: es
el único sitio donde conoces la respuesta de antemano, así que te da las dos referencias —frase presente
y frase ausente— contra las que comparar el 0,480 del test.

---

*Las cifras de este guion son las mismas que las del mazo y las del paper; su procedencia está
documentada en `deliverables/paper/FACTS.md`.*
