# Computer Can Do Art

The assignment is solved by me, Artem Bakhanov, a student of Innopolis University. If you have any question regarding any part of this document and other provided materials, you can contact me via email: [a.bahanov@innopolis.university](mailto:a.bahanov@innopolis.university)

In this assignment, I used Python 3.7 as a programming language. Also OpenCV and Numpy were used to work with images easily and with high performance. For generating Voronoi polygons I used Scipy. All files with code are in directory [/code](/code). 

You can also read my [report](report.pdf).

<iframe src='https://gfycat.com/ifr/TangiblePessimisticArrowworm' frameborder='0' scrolling='no' allowfullscreen width='640' height='684'></iframe>

## Running the code

First of all, you need to install **Python** of version 3.7. Note, that the code was not tested on Python 3.8, I cannot guarantee it works on Python 3.8. Install dependencies with `pip install -r requirements.txt` (this command may vary on different OSs and computers). Then you can run `main.py` and it will generate some image. If you want another image or you want to change parameters - edit `main.py` file (it is not good practice; I will fix it soon).

 ## The story

In the beginning, I had a lot of ideas which were about generating triangles or more complex polygons to recreate the original image. Initially, I created a program that did it, but I did not like the result I got. I started researching and probing a lot. For now, I created about six different algorithms, and I was not satisfied with any of them. The problem was that it was too edgy and "sharp". The images I got were just inaccurate replica of the original images. So I decided that randomness in the figures is not cool, and I remembered my internship in 2019 when I was creating an app that reconstructs 3d mesh. One of the tasks that I needed to solve is Voronoi Partition. This algorithm gives wonderful images. I took the algorithm, and I took one my idea from the previous algorithm - RGB splitting. My final program creates an image out of 3 independent Voronoi Diagrams - each in a separate color layer. And the result is interesting. It is structured and chaotic at the same time.

## Image Examples

<img src="/voronoi_output/voronoi35.png"  />

![fefe](/voronoi_output/voronoi12.png)

<img src="C:\Users\artem\Code\ComputerCanDoArt\voronoi_output\voronoi21.png" style="zoom: 67%;" /> <img src="/voronoi_output/voronoi23.png" style="zoom:67%;" />

## Population

This paragraph is about the size of the population. I tested my algorithm on a big number of tests, and I got to a conclusion: there is no need to generate more than one entity in one population. I tried to change a lot of parameters but it did not change the overall image, and I can explain why it happens. My image is generated using polygons, and the number of polygons is plus or minus constant (the number can change because the Voronoi polygons can create polygons that are impossible to draw). For this reason, every time we will get different images in the genome but they will look very similar for a human eye. Yes, the genotype is different but for humans only phenotype is important.
Another problem was convergence of the population to one good result. I tried different techniques described in the following paragraphs but it did not yield any improvements. For this reason, I did not use any crossover function. Every iteration (generation) two mutated kids are generated and selection algorithm is applied. The artistic results became better due to this improvement. Below you can see three images. two of them were generated using only mutations and one with population size = 10 and 2 point crossover. The results (phenotypes) are very similar. Since I do not use any crossover function, I do not use parent selection algorithm. But I tried panmixis and random outbreeding. Random outbreeding was more effective in terms of generating diverse genotypes. But eventually, the diversity of the genotype was really small (on a big number of populations).

## Fitness Function

I use is inverse of Mean Squared Error.

## Acknowledgements

I am very thankful to my professor of AI, Josepf Brown for giving very comprehensive Introduction to AI classes.