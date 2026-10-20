# Recuperacion de citas: prueba aislada de frutos

La correccion se incorporo posteriormente a los datos procesados y tablas
canonicas. Ver [documentacion de la revision](../../../docs/fruit_citation_revision.md).
Este directorio conserva el antes/despues y reproduce la prueba desde
`baseline_candidates.csv`, no desde las tablas canonicas ya corregidas.

## Resultado

Los tres frutos seleccionados no cambian ni cambian de orden. Sus puntajes
si cambian porque se recalculan las normalizaciones sobre los 120 candidatos.
El conjunto underabsorbed sigue teniendo 11 candidatos.

| Fruto | Citas externas | Indegree local | Puntaje original | Recalculado |
| --- | ---: | ---: | ---: | ---: |
| Sulaiman et al. (2024) | 26 | 0 | 0.717107 | 0.667759 |
| Kakeesh (2025) | 8 | 0 | 0.554811 | 0.520196 |
| Hanaysha (2022) | 11 | 0 | 0.542263 | 0.514742 |

Los nombres abreviados remiten a los registros completos de top3_recalculated.csv.

## Diez registros recuperados

Todos tenian cited_by ausente en la tabla integrada, convertido a cero por
el algoritmo. Los valores siguientes se recuperaron por coincidencia exacta
de DOI, ignorando mayusculas, con un unico registro en el archivo Scopus
[EM 7MAR26 243.csv](https://drive.google.com/file/d/13gnqMhYk2iXAbjdELTns_df9wDmzgpMO/view).
Son cifras de la exportacion del 7 de marzo de 2026, no consultas actuales.

| Registro | Citas recuperadas | Indegree local | Puntaje recalculado | Clasificacion |
| --- | ---: | ---: | ---: | --- |
| Dhameria (2021) | 20 | 4 | 0.330436 | fruit_candidate |
| Al-Shaikh (2023) | 37 | 5 | 0.421247 | fruit_candidate |
| Saiyed (2023) | 10 | 2 | 0.283482 | fruit_candidate |
| Guerola-Navarro (2024) | 80 | 4 | 0.575375 | fruit_candidate |
| Polas (2021) | 65 | 3 | 0.487529 | fruit_candidate |
| Nguyen (2021) | 14 | No emparejado | 0.418803 | external_signal_not_matched_in_graph |
| Sun (2022) | 15 | 7 | 0.203179 | already_absorbing_into_tree |
| Siregar (2024) | 4 | No emparejado | 0.255633 | external_signal_not_matched_in_graph |
| Mugoni (2023) | 63 | No emparejado | 0.672866 | external_signal_not_matched_in_graph |
| Saki (2023) | 2 | No emparejado | 0.150798 | external_signal_not_matched_in_graph |

Mugoni supera el puntaje de Sulaiman, pero no es elegible como fruto
underabsorbed por no estar emparejado con la red. El cero almacenado como
indegree para los cuatro no emparejados es un valor por defecto, no un grado
observado. No se investigaron ni corrigieron sus emparejamientos en esta prueba.
Los otros seis tienen indegree mayor que uno y tampoco cumplen la regla
underabsorbed. Un puntaje alto, por si solo, no determina la seleccion final.

## Alcance y verificacion

- Se conserva la poblacion guardada de 120 candidatos, el anio 2026, las
  caracteristicas de red y texto y los pesos 0.35/0.35/0.20/0.10.
- Se usan las funciones originales normalize_series y classify_fruit_signal.
  La puntuacion replica las operaciones de src/fruit_candidates.py.
- Antes de recuperar las citas se reprodujeron los puntajes originales con
  error maximo 3.34e-16, las categorias exactas y los tres frutos guardados.
- Se verifico que las diez cifras estaban realmente ausentes en la tabla
  integrada; no se sustituyeron ceros reportados por una fuente.
- Se recalcularon velocidad, indicadores normalizados, brecha, puntuacion
  y categorias sobre los 120 registros. No se sumaron citas entre fuentes.
- Persisten las diferencias de cobertura entre WoS Z9 y Scopus Cited by.
  Esta prueba no homogeneiza fuentes, reconstruye la red ni valida relevancia.
- En la prueba inicial se comprobaron 137 archivos originales mediante SHA-256
  sin modificarlos. La reproduccion actual verifica que los cinco archivos
  canonicos de entrada/salida de frutos permanezcan intactos durante la prueba;
  estos ya contienen las correcciones incorporadas posteriormente.

## Archivos

- data/corrections/fruit_citation_recovery_2026.json (desde la raiz del repo):
  diez correcciones con DOI y procedencia, compartidas con el pipeline.
- baseline_candidates.csv: poblacion original antes de corregir los conteos.
- comparison_recovered10.csv: antes/despues de los diez candidatos.
- comparison_all120.csv: comparacion de toda la poblacion y posiciones
  entre los candidatos underabsorbed (vacias para los no elegibles).
- candidates_recalculated.csv: todas las variables de los 120 candidatos.
- underabsorbed_recalculated.csv y top3_recalculated.csv: seleccion experimental.
- audit_manifest.json: parametros, comprobaciones, huella del baseline y
  huellas de las tablas canonicas al ejecutar la prueba.
- recalculate.py: reproduccion aislada; ejecutarlo desde cualquier carpeta
  con un Python que tenga las dependencias de investigacion del repositorio.

La estabilidad de estos tres nombres solo queda demostrada para esta
correccion acotada, no para cambios de fuentes, emparejamientos o algoritmo.
