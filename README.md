# FedMutSigExtract
This project applies Federated Learning (FL) to extract mutational signatures from genomic data in a privacy-preserving manner. By utilizing Non-negative Matrix Factorization (NMF) and autoencoders, the system enables collaborative analysis of decentralized healthcare data without sharing sensitive patient information. The FL approach is tested on synthetic and real-world genomic datasets to evaluate its accuracy and computational efficiency compared to traditional centralized methods.

# Installation
Clone the project using the following command

```bash
git clone https://github.com/dunkedolmer/FedMutSigExtract.git
```

Ensure that you have the newest version of Python installed on your system.

Navigate to the project directory:
```
cd project_repository
```

Create a virtual environment (optional but recommended):

```
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# Unix or MacOS
source venv/bin/activate
```

Next, install the necessary dependencies for this project:

```bash
pip install -r requirements.txt
```

This will install the required Python packages specified in the requirements.txt file.

# Example

To run the system with the federated autoencoder, you need to do the following:

### 1. Prepare the Setting

Open four terminals (preferably in your IDE) to run one instance of the central server and three instances of clients.

### 2. Start the Central Server

Navigate to the `root/src` directory and run the server using Python:

```bash
cd root/src
python server.py
```

The server now listens for participating clients and starts the federated learning process once three clients participate (e.g., when three clients are running)

### 3. Start the Clients

For the clients, navigate to the `root/src/federated-deepms-autoencoder` directory and run the client script with the partition (either 1, 2, or 3) using Python. You need to run three separate clients, each with a different partition number.

For partition 1:
```bash
cd root/src/federated-deepms-autoencoder
python client.py 1
```

For partition 2:
```bash
cd root/src/federated-deepms-autoencoder
python client.py 2
```

For partition 3:
```bash
cd root/src/federated-deepms-autoencoder
python client.py 3
```

---

# Contributors
Frederik Rasmussen - frasm19@student.aau.dk

Kevin Risgaard Sinding - ksindi19@student.aau.dk
