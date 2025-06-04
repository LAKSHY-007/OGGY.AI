{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "197e7fa5",
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "import torch\n",
    "\n",
    "print(\"Number of GPU: \", torch.cuda.device_count())\n",
    "print(\"GPU Name: \", torch.cuda.get_device_name())"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
