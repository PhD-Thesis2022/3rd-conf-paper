This is the python code of the paper "Detect & Reject for Transferability of Black-box Adversarial Attacks Against Network Intrusion Detection Systems"

use this file : "Same Training - Transferability property of adversarial attacks against intrusion detection systems.py"  to test the effect of the transferability property of adversarial attacks against intrusion detection systems when both IDSs are trained with the same. 

If you want to train each IDS with a diferent dataset, use this file: "Different Training - Transferability property of adversarial attacks against intrusion detection systems.py"

Finally, to test the defense called "Detect & Reject" against the transferability of adversarial attacks use the file: "Detect and Reject Defense - Transferability property of adversarial attacks against intrusion detection systems" 

You will need the NSL-KDD Dataset to run these codes.
https://www.unb.ca/cic/datasets/nsl.html

Adversarial Robustness Toolbox (ART) - Python Library for Machine Learning Security - Evasion, Poisoning, Extraction, Inference was used for the tests.
https://github.com/Trusted-AI/adversarial-robustness-toolbox

TensorFlow is used for building a DNN-based IDS.
https://www.tensorflow.org/

Scikit-learn is used to build five Machine learning-base IDSs. 
https://scikit-learn.org/
