# CNN-based Relation Extraction

## About

CNN-based Relation Extraction Model

## prerequisite
* `java 1.8`
* `maven`
* Korean word embedding file [download here](https://drive.google.com/file/d/1UwCp0xwfgl9185B_iJ2ZY3W7vSEsqLu0/view?usp=sharing)

## How to use
refer `edu.kaist.mrlab.nn.pcnn.pipeline` package<br>
  * `Learning.java`
  * `Extraction.java`

### How to run
`mvn clean compile exec:java`

## Licenses
* `CC BY-NC-SA` [Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/2.0/)
* If you want to commercialize this resource, [please contact to us](http://mrlab.kaist.ac.kr/contact)

## Maintainer
Sangha Nam `nam.sangha@kaist.ac.kr`

## Publisher
[Machine Reading Lab](http://mrlab.kaist.ac.kr/) @ KAIST

## Citation
@article{nam2018distant,
  title={Distant Supervision for Relation Extraction with Multi-sense Word Embedding},
  author={Nam, Sangha and Han, Kijong and Kim, Eun-kyung and Choi, Key-Sun},
  journal={Global Wordnet Conference, Workshops on Wordnets and Word Embeddings},
  year={2018}
}
