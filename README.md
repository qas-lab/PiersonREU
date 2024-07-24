# Automated Bug Report Prioritization in Large Open-Source Projects

There are multiple automated approaches to bug triage, though many have proven less effective than the manual alternative. These approaches are convenient for detecting duplicate bugs and assigning bugs to various triagers depending on their general problem topics. However, automated approaches tend to lack the ability to categorize bug priority from bug reports accurately. In this study, we approach the prioritization dilemma through a novel approach to classify bugs into appropriate categories using a multi-faceted technique using recent advancements in Natural Language Processing (NLP). To this aim, we analyze the natural language used within bug reports, using topic modeling and text classification to prioritize bug reports properly.

## Installation

This approach uses Jupyter Notebooks and Python for implementation. You can access these files in Jupyter Notebooks by git cloning the repository.

## Usage

If you are planning to implement the entire process we used:

1) InitialProcessing: To upload data.

2) TopicModeling: Creating our topics using LDA with the dataset.

3) TrainingClassifiers: We separated this folder into BERT and Naive Bayes depending on which text classifier you plan to use. Once you have your LDA topics, you train a text classifier for each topic using these codes.

4) Pipelines: Once you have your topics and text classifiers, you may implement the pipelines accordingly. To test a dataset using this repository, you must access the pipelines. To test your dataset with any of the three pipelines, make sure your priority levels are organized using label_map = {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4, 'P5': 5}. From there, it is easy to compare the pipeline and actual results.

If implementing this process from scratch, you may have to adjust the file names within these codes.

## Credits

[1] Q. Umer, H. Liu, and Y. Sultan, “Emotion based automated priority
prediction for bug reports,” IEEE Access, vol. PP, pp. 1–1, 07 2018.
[2] “How to set severity and priority,” https://wiki.eclipse.org/WTP/
Conventions of bug priority and severity#:∼:text=Severity%20is%
20assigned%20by%20a,priority%2C%20P5%20is%20the%20lowest.,
accessed: 2024-05-28.
[3] J. Anvik, L. Hiew, and G. C. Murphy, “Who should fix this bug?”
in Proceedings of the 28th International Conference on Software
Engineering, ser. ICSE ’06. New York, NY, USA: Association
for Computing Machinery, 2006, p. 361–370. [Online]. Available:
https://doi.org/10.1145/1134285.1134336
[4] G. Jeong, S. Kim, and T. Zimmermann, “Improving bug triage with bug
tossing graphs,” in Proceedings of the 7th Joint Meeting of the European
Software Engineering Conference and the ACM SIGSOFT Symposium
on The Foundations of Software Engineering, ser. ESEC/FSE ’09.
New York, NY, USA: Association for Computing Machinery, 2009, p.
111–120. [Online]. Available: https://doi.org/10.1145/1595696.1595715
[5] C. Sun, D. Lo, S.-C. Khoo, and J. Jiang, “Towards more accurate
retrieval of duplicate bug reports,” in 2011 26th IEEE/ACM International
Conference on Automated Software Engineering (ASE 2011), 2011, pp.
253–262.
[6] X. Xia, D. Lo, Y. Ding, J. M. Al-Kofahi, T. N. Nguyen, and X. Wang,
“Improving automated bug triaging with specialized topic model,” IEEE
Transactions on Software Engineering, vol. 43, no. 3, pp. 272–297, 2017.
[7] A. Ali, Y. Xia, Q. Umer, and M. Osman, “Bert based severity prediction
of bug reports for the maintenance of mobile applications,” Journal of
Systems and Software, vol. 208, p. 111898, 2024. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S0164121223002935
[8] J. Anvik, L. Hiew, and G. C. Murphy, “Coping with an open
bug repository,” in Proceedings of the 2005 OOPSLA Workshop on
Eclipse Technology EXchange, ser. eclipse ’05. New York, NY,
USA: Association for Computing Machinery, 2005, p. 35–39. [Online].
Available: https://doi.org/10.1145/1117696.1117704
[9] W. Zhang and C. Challis, “Software component prediction for bug
reports,” in Proceedings of The Eleventh Asian Conference on Machine
Learning, ser. Proceedings of Machine Learning Research, W. S. Lee
and T. Suzuki, Eds., vol. 101. PMLR, 17–19 Nov 2019, pp. 806–821.
[Online]. Available: https://proceedings.mlr.press/v101/zhang19c.html
[10] K. R. Chowdhary, Natural Language Processing. New Delhi:
Springer India, 2020, pp. 603–649. [Online]. Available: https:
//doi.org/10.1007/978-81-322-3972-7 19
[11] Y. Tian, D. Lo, X. Xia, and C. Sun, “Automated prediction of
bug report priority using multi-factor analysis,” Empirical Softw.
Engg., vol. 20, no. 5, p. 1354–1383, oct 2015. [Online]. Available:
https://doi.org/10.1007/s10664-014-9331-y
[12] D. M. Blei, A. Y. Ng, and M. I. Jordan, “Latent dirichlet allocation,”
Journal of machine Learning research, vol. 3, no. Jan, pp. 993–1022,
2003.
[13] X. Wu, T. Nguyen, and A. T. Luu, “A survey on neural
topic models: methods, applications, and challenges,” Artificial
Intelligence Review, vol. 57, no. 18, 2024. [Online]. Available:
https://doi.org/10.1007/s10462-023-10661-7
[14] A. Yadav and S. S. Rathore, “A hierarchical attention networks
based model for bug report prioritization,” in Proceedings of the 17th
Innovations in Software Engineering Conference, ser. ISEC ’24. New
York, NY, USA: Association for Computing Machinery, 2024. [Online].
Available: https://doi.org/10.1145/3641399.3641416
[15] G. Weiss, Foundations of Imbalanced Learning, 06 2013, pp. 13–41.
[16] “Eclipseplatform,” 2018. [Online]. Available: https://github.com/logpai/
bughub/tree/master/EclipsePlatform
[17] “bughub,” 2018. [Online]. Available: https://github.com/logpai/bughub

