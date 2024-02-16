# Defect-Detection

## Введение 
Область детектирования изображений с использованием глубокого обучения постоянно развивается – появляется множество новых техник и моделей, которые находят применение в промышленном производстве. Одной из таких возможных областей применения является детектирование дефектов в фоторезистивной маске на SEM-изображениях тестовых структур. Использование традиционных методов обнаружения дефектов требует значительных затрат времени и средств, так как эти методы основаны на ручном анализе изображений и требуют высокой квалификации специалистов. Данные методы имеют определённые ограничения, поскольку выполняемая классификация дефектов основана на заранее заложенных правилах. Эти ограничения часто приводят к неправильной классификации дефектов, тем самым увеличивая время анализа снимков. Кроме того, эти методы не позволяют обнаруживать и классифицировать новые типы дефектов, возникающих при переходе к более низким технологическим нормам. Возможное наличие шума на SEM-изображениях также негативно влияет на корректность результата детектирования. Использование нейросетевых детекторов позволяет автоматизировать процесс детекции дефектов при анализе больших наборов изображений, обеспечивает устойчивость к шуму и возможность обнаружения новых дефектных структур. Работа направлена на то, чтобы продемонстрировать возможность методов глубокого обучения точно классифицировать и локализовать различные типы дефектов фоторезистивной маски. 
В данной работе рассмотрены три популярные модели нейросетевых детекторов: `YoloV5`, `Faster R-CNN`, `RT-DETR`. Каждая из них имеет свои особенности и преимущества, которые будут подробно описаны в статье. Также будут приведены результаты экспериментов на собственном наборе данных, демонстрирующие эффективность каждой из моделей. Для оценки точности предсказаний использовались метрики **mAP@0.5** и **mAP@0.5-0.95**. Скорость детекторов оценивалась с помощью среднего числа кадров в секунду(**FPS**) во время инференса на тестовом наборе данных. 

## Датасет
Для создания датасета использовались изображения дефектов фоторезистивной маски в затворном слое для проектной нормы **28нм**.  Датасет содержит 500 RGB изображений формата “.tiff” размером **1024x1024 px**. При анализе имеющихся снимков удалось выделить 3 типа дефектных структур: **SRAF**(от англ. sub-resolution assist features) – вспомогательные непечатаемые структуры, добавляемые на кремниевую пластину для расширения окна процесса , **BRIDGE** – слияние двух соседних структур , **GAP** – промежуток в пределах одной структуры.
Пример каждого типа дефектных структур отражён на рисунке 1:
<img width="653" alt="Снимок экрана 2023-12-21 в 11 43 19" src="https://github.com/mskv99/Defect-Detection/assets/119602773/3cd965ea-1474-4db3-a10b-b71cc94c318f">



Каждый тип дефекта представлен 150 изображениями. Оставшиеся 50 изображений представляют собой структуры, не содержащие дефектов. Набор данных был разделён на тренировочную, валидационную и тестовую выборки в соотношении `76:17:7(380:83:37)`. В таблице 1 приведена дополнительная  статистика по датасету, включая количество изображений и количество структур для каждой из выборок. 

|        |Train|Validation|Test|
|:--------|:-----:|:----------:|:----:|
|  **GAP**   | 725 |	  85    |	53 |
| **SRAF**   |426  |	126     |	51 |
|**BRIDGE**  |348  |	79      |	45 |
|**Число структур** |	1499 |	290 |	149 |
|**Число изображений** |	380 |	83 |	37 |

**Таблица 1**: Статистика по датасету SEM-изображений для каждой из выборок




## Эксперимент
Для обучения использовались три модели детекторов: **Faster R-CNN**, **YoloV5**, **RT-DETR**. В первой модели детектора использовались предобученные **ResNet50-FPN-v2**, **MobileNetV3-Large-FPN**, **ResNet101** в качестве сетей извлечения признаков. Для подбора гиперпараметров модели использовался алгоритм случайного поиска: подбирались тип оптимизатора, размер батча, learning rate, scheduler; дообучение внутри случайного поиска выполнялось на 5 эпохах для тестирования. Обучение проводилось на 40 эпохах.

В экспериментах с **YoloV5** тестировались 3 предобученные версии модели с различным количеством весов (**s** - малая, **m** - средняя, **l** - большая) со стандартным набором гиперпараметров. Обучение проводилось на 50 эпохах. Для моделей, показавших наилучший результат был дополнительно проведён подбор гиперпараметров с помощью встроенного генетического алгоритма и проведены повторные эксперименты. Дообучение внутри генетического алгоритма проводилось на 10 эпохах для тестирования.

В работе с **RT-DETR** были выбраны модели предобученные модели **RT-DETR-l** и **RT-DETR-x** с сетью извлечения признаков **HGNetv2**. Для этой серии был выбран стандартный набор гиперпараметров, за исключением лишь типа оптимизатора и параметров аугментации, которые были выбраны совпадающими с аналогичными у YoloV5 для более последовательного сравнения. Размер батча также пришлось понизить c 16 до 10 для более оптимального процесса обучения. Обучение для RT-DETR проводилось на 50 эпохах.



## Графики обучения
* [FasterRCNN](https://api.wandb.ai/links/ml_team_mskv/ct39z2bb)
* [YoloV5](https://wandb.ai/ml_team_mskv/YOLOv5/reports/YoloV5--Vmlldzo1MTA2OTMw?accessToken=z32xznnqlqqtrt1i3xuqhu4bimcaeh0ys879jus351ugovp3un8lximih0e038kx)
* [RT-DETR](https://www.comet.com/mldrdrew99/rt-detr/reports/rt-detr)
## Результаты
*Результаты в ближайшее время будут уточнены с учётом величин стандартных отклонений и нового эксперимента с моделью Yolo-NAS*

|                                          |  SRAF | BRIDGE |  GAP  | All(mAP@0.5) |
|:----------------------------------------:|:-----:|:------:|:-----:|:------------:|
|      **Faster R-CNN** (ResNet50-FPN-v2)      | 0.948 |  0.964 | 0.782 |     0.898    |
|   **Faster R-CNN** (MobileNetV3-Large-FPN)   | 0.919 |  0.953 | 0.690 |     0.854    |
|         **Faster R-CNN** (ResNet101)         | 0.920 |  0.977 | 0.689 |     0.862    |
|                  **YoloV5s**                 | 0.958 |  0.979 | 0.806 |     0.914    |
|            **YoloV5s** (optimized)           | 0.963 |  0.985 | 0.786 |     0.911    |
|                  **YoloV5m**                 | 0.973 |  0.983 | 0.809 |     0.921    |
|           **YoloV5m**  (optimized)           | 0.971 |  0.983 | **0.830** |     **0.928**    |
|                  **YoloV5l**                 | **0.981** |  0.977 | 0.788 |     0.915    |
|            **YoloV5l** (optimized)           | 0.964 |  **0.986** | 0.743 |     0.898    |
| **Ensemble YoloV5(s/m/l)** (default/opt/opt) | 0.967 |  0.983 | 0.828 |     0.926    |
|                 **RT-DETR-l**                | 0.967 |  0.977 |  0.77 |     0.904    |
|                 **RT-DETR-x**                | 0.969 |  0.985 | 0.769 |     0.907    |

**Таблица 2**: точность по каждому классу и по всем классам при уровне порога **IoU=0.5** для валидационной выборки


|                                          |  SRAF | BRIDGE |  GAP  | All(mAP@0.5:0.95) |
|:----------------------------------------:|:-----:|:------:|:-----:|:-----------------:|
|      **Faster R-CNN** (ResNet50-FPN-v2)      | 0.662 |  0.544 | 0.403 |       0.536       |
|   **Faster R-CNN** (MobileNetV3-Large-FPN)   | 0.610 |  0.514 | 0.337 |       0.487       |
|         **Faster R-CNN** (ResNet101)         | 0.610 |  0.468 | 0.268 |       0.449       |
|                  **YoloV5s**                 | 0.609 |  0.555 | **0.438** |       0.534       |
|            **YoloV5s** (optimized)           | 0.621 |  0.562 | 0.393 |       0.525       |
|                  **YoloV5m**                 | 0.603 |  0.544 | 0.432 |       0.526       |
|           **YoloV5m**  (optimized)           | 0.653 |  0.579 | 0.435 |       **0.556**       |
|                  **YoloV5l**                 | 0.670 |  0.546 | 0.389 |       0.535       |
|            **YoloV5l** (optimized)           | 0.663 |  **0.593** |  0.38 |       0.545       |
| **Ensemble YoloV5(s/m/l)** (default/opt/opt) | 0.650 |  0.571 | 0.435 |       0.552       |
|                 **RT-DETR-l**                | **0.689** |  0.559 | 0.393 |       0.547       |
|                 **RT-DETR-x**                | 0.638 |  0.555 | 0.396 |        0.53       |

**Таблица 3**: точность по каждому классу и по всем классам при различных уровнях порога **IoU 0.5:0.95** для валидационной выборки


|                                      |  FPS  |
|:------------------------------------:|:-----:|
|    **Faster R-CNN**  (Resnet50-FPN-v2)   |  8.76 |
| **Faster R-CNN** (MobileNetV3-Large-FPN) | 31.18 |
|       **Faster R-CNN** (ResNet101)       |  8.32 |
|           **YoloV5s** (default)          | 58.14 |
|          **YoloV5m** (optimized)         | 32.05 |
|          **YoloV5l** (optimized)         | 25.84 |
|        **Ensemble YoloV5**(s/m/l)        | 17.09 |
|               **RT-DETR-l**              | 27.78 |
|               **RT-DETR-x**              | 25.13 |

**Таблица 4**: FPS для различных конфигураций моделей детекторов на  Tesla T4 GPU


## Веса моделей
|             model name                   |
|:----------------------------------------:|
| [**Faster R-CNN**  (Resnet50-FPN-v2)](https://drive.google.com/file/d/1-dLBgpC4e4PL9Xl84aRA8zkWnshp1arm/view?usp=sharing)   | 
| [**Faster R-CNN** (MobileNetV3-Large-FPN)](https://drive.google.com/file/d/1-AuW0iwxTGy96HYhCn3nWFN61H_h0kO4/view?usp=sharing) | 
| [**YoloV5s** (default)](https://drive.google.com/file/d/1-19iOTXsu9_dltyM_-m6g9L8DbP4mwZJ/view?usp=sharing)          | 
| [**YoloV5m** (optimized)](https://drive.google.com/file/d/1-rsaF5_9r_-3tQURl1Ubr86M0k883g8K/view?usp=sharing)         | 
| [**YoloV5l** (optimized)](https://drive.google.com/file/d/1-snnfjWWDrZNZLwNW2VG3VV-sQJFi7dj/view?usp=sharing)         | 
| [**RT-DETR-l**](https://drive.google.com/file/d/1-0v-BSLm2acIgzN31A8Df2l3acGUN_mz/view?usp=sharing)              | 
| [**RT-DETR-x**](https://drive.google.com/file/d/1-1z8KREGyzb1jNTK_8srFuANJNWroijR/view?usp=sharing)              |
|[**Yolo-NAS-s**](https://drive.google.com/file/d/1R1kCsbbr8efXn6X-E0RcFi94hmCpNnR3/view?usp=sharing)   |
|[**Yolo-NAS-m**](https://drive.google.com/file/d/1--ns3JQQ4KYO-BWGC7Qv8ILR1liTjLXM/view?usp=sharing)|
|[**Yolo-NAS-l**](https://drive.google.com/file/d/1-36Jo-noUxDncl4s7do8RAy7iwtFIZDM/view?usp=sharing)|




## Вывод

*Выводы в ближайшее время будут уточнены с учётом величин стандартных отклонений и нового эксперимента с моделью Yolo-NAS*
В работе проведён сравнительный анализ моделей нейросетевых детекторов. Все модели детекторов продемонстрировали высокую среднюю точность детекции по всем классам как при единственном значении порога IoU(**mAP@0.5**≥0.854), так и при нескольких пороговых значениях IoU(**mAP@0.5:0.95** > 0.449).

Наиболее сложным классом для детекции оказался дефект типа’GAP’. Данный тип дефекта не всегда просто отличить от фона на изображении или от самой структуры, на которой он присутствует. При составлении набора данных необходимо также учитывать, что количество структур данного типа на одном изображении в среднем превосходит количество дефектов других типов. Это важно для создания сбалансированного набора данных.

Эксперимент по подбору гиперпараметров  для модели дал положительный результат для средней модели YoloV5-m, в результате чего выросла точность по каждому из классов в отдельности и общая точность. В случае с малой YoloV5-s и большой YoloV5-l моделями подбор гиперпараметров приводил к росту точности для отдельных классов дефектов.

Ансамбль из стандартной YoloV5-s и оптимизированных YoloV5-m и YoloV5-l увеличил общий **mAP@0.5** и **mAP@0.5:0.95** для малой и большой моделей, но снизил данный показатель для средней модели. 

Модель YoloV5-m имеет наибольшие показатели точности по всем классам(0.928 и 0.556). Наиболее предпочтительными для использования по мнению авторов являются оптимизированная средняя модель и стандартная малая модель. 

Наибольший FPS имеют малая(**58.14**) и средняя(**32.05**) модели YoloV5.

В дальнейшем планируется расширение набора возможных дефектов на SEM-изображениях.

