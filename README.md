# 1 Package Description
This repository provides supplementary material to the ICSE 2023 paper "Analyzing the Impact of Workloads on Modeling the Performance of Configurable Software Systems", including experimental data (configuration samples, subject system code) and measurements (performance, option- and workload-specific code coverage) as well as an interactive dashboard. The latter complements the presentation of the study results in the paper and allows for reproduction of our analyses and findings.

## 1.1 Experimental Data

The material includes information about the following nine software systems:

| Name  |  Domain |  Language | Repository  |  Code Used / Version  | License |
|---|---|---|---|---|---|
|  `jump3r` |  Audio Encoder |  Java |  [`Link`](https://github.com/Sciss/jump3r) | [`v1.0.4`](misc/code/jump3r)  | LGPL 2.1 |
|  `kanzi` |  File Compressor  | Java  | [`Link`](https://github.com/flanglet/kanzsample.csv)  | [`v1.9`](misc/code/kanzi)  | Apache License 2.0 |
|  `dconvert` |  Image Scaling |  Java | [`Link`](https://github.com/patrickfav/density-converter) | [`v1.0.0.-alpha7`](misc/code/dconvert) | Apache License 2.0 |
|  `h2` | Database  |  Java | [`Link`](https://github.com/h2database/h2database)  |  [`v1.4.200`](misc/code/h2) | Mozilla Public License version 2.0 |
|  `batik` |  SVG Rasterizer |  Java | [`Link`](https://github.com/apache/xmlgraphics-batik)  |  [`v.1.14`](misc/code/batik) |  Apache License 2.0 | 
|  `xz` | File Compessor  |  C/C++ | [`Link`](https://github.com/xz-mirror/xz)  | [`v5.2.0`](misc/code/xz)  |  GPL 2.0 and GPL 3.0|
|  `lrzip` |  File Compressor | C/C++  |  [`Link`](https://github.com/ckolivas/lrzip) |  [`v0.651`](misc/code/lrzip) | GPL 2.0 |
|  `x264` |  Video Encoder | C/C++  |  [`Link`](https://github.com/mirror/x264) |  [`baee400..`](misc/code/x264) | GPL 2.0 |
|  `z3` | SMT Solver  |  C/C++ |  [`Link`](https://github.com/Z3Prover/z3) | [`v4.8.14`](misc/code/z3)  | MIT License |

### 1.1.1 Configurations
This archive includes the configurations sampled and used for conducting the experiments as CSV files:

	data/coverage_raw.tar.gz

### 1.1.2 Workloads
The files used as workloads/inputs in the experiments were collected from various sources. We provide a list of the provenance of 
the files at:

	data/workload_sources.csv

In compliance with the license (LICENSE.txt). This archive does not include the complete set of files used because not all 
files used allow for redistribution under the CC BY-SA 4.0 license.

### 1.1.3 Performance Measurements
This archive includes the performance measurements (throughput and execution time) per configuration and workload as CSV files.
The performance measurements for each `<software system>` can be found in the following subfolder:

	dashboard/resources/<software_system>/measurements.csv

### 1.1.4 Coverage Measurements
This archive includes (as a separate file!) raw coverge measurements per configuration and workload as CSV files. Based on these files, the option- and
 workload-specific code is inferred. The raw code coverage measuremnts can be found as a gz-ipped tar archive:

	./coverage_raw.tar.gz

### 1.1.5 Option-specific Code
This archive includes the calculated code sections that are a) option-specific and b) workload- and option-specific.
The code sections for each `<software system>` can be found in the following subfolder:

	dashboard/resources/<software_system>/code/

## 1.2 Interactive Dashboard
We provide an interactive dashboard using the frameword `streamlit` that allows exploring in 
detail our analysis for each configuration option and workload. The original paper presents a representative subset of
the results, all analyses and visualizations can be reproduced using this dashboard. We provide a Docker-ized 
version to run the dashboard locally.

To build and run the Docker container execute the following commands:

```
#!/bin/sh
cd ./dashboard
docker build -t streamlitapp:latest .
docker run -p 8501:8501 streamlitapp:latest
```

You can now explore the dashboard running locally at https://127.0.0.1:8501 or https://localhost:8501.

To use the dockerized application, the container environment Docker has to be set up and running. To install
Docker, these tutorials provide orientation: 

* Windows: https://docs.docker.com/desktop/install/windows-install/
* Mac https://docs.docker.com/desktop/install/mac-install/
* Linux: https://docs.docker.com/desktop/install/windows-install/
