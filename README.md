<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->


[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/kallebylin/repo_name">
    <img src="images/logo_hse.jpg" alt="Logo" height="80">
  </a>

  <h3 align="center">Higher School of Economics Large Scale ML Final Project</h3>

  <p align="center">
    This project implements the Temporal Fusion Transformer architecture for interpretable multi-horizon forecasting (Lim et al., 2020). It is a modern attention-based architecture designed for dealing with high-dimensional time series with multiple inputs, missing values and irregular timestamps.
    <br />
    <a href="https://arxiv.org/pdf/1912.09363.pdf"><strong>See original paper »</strong></a>
    <br />
    <br />
    <a href="https://github.com/KalleBylin/tft_webapp/issues">Report Bug</a>
    ·
    <a href="https://github.com/KalleBylin/tft_webapp/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![product-screenshot](images/webapp.png)

Machine learning is today widely used around the world and has opened up new possibilities to extract deep insights from data and allow machines to make high quality predictions. Still, one of the largest challenges today in ML has to do with the deployment of these algorithms with large scale data. 

Large scale Machine Learning aims to solve these challenges through tools and methods like model optimization, computation parallelism, and scalable deployment.

This project is an example on how a deep learning model can be deployed to a webapp with asynchronous request processing. 


### Built With

* [FastAPI](https://fastapi.tiangolo.com/)
* [Pytorch](https://pytorch.org/)
* [Celery](https://docs.celeryproject.org/en/stable/index.html)
* [Redis](https://redis.io/)
* [Docker](https://www.docker.com/)
* [Dash](https://plotly.com/dash/open-source/)


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/KalleBylin/tft_webapp.git
   ```

This will clone the latest version of the TFT Webapp repository to your machine.

2. Install docker and docker-compose

Make sure you have installed Docker on your machines in order to run this project.


###### Run entire app with one command 
```
sudo docker-compose up --build
```

This will start multiple services:

* Sets up a Redis database whcih will be used as a message broker and keep track of the tasks' states.
* Initializes a Celery app which acts as our task queue so that they can run asynchronously.
* Starts a model web server with rest api built with FastAPI and listens for messages at localhost:8000. 
* Deploys a web app that can be opened in the browser and used to upload a batch of data and then receives predictions from the model web server.


#### Test over REST API

We can send a sample of data to the model through a POST request like this:

```python
r = requests.post("http://model_server:8000/predict", json=json_data)
```

Here json_data corresponds to the inputs of the model in JSON format. 

This POST request will return the task_id of the prediction task that is requested:

```
{
  "task_id": "353286k1-j125-6776-9889-f7b447nat1fcb"
}
```

The task will be handled by Celery. In the meantime we can send a GET request to understand the status of the task:

```python
r = requests.get("http://model_server:8000/predict/353286k1-j125-6776-9889-f7b447nat1fcb")
```

This will either give us an update stating that the task is in progress:

```
{
  "status": "IN_PROGRESS"
}
```

Or return a status of "DONE" with the output of the task:

```
{
  "status": "DONE",
  "result": {
    "outputs": [[...]]
  }
}
```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/kallebylin/repo_name/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email

Project Link: [https://github.com/kallebylin/repo_name](https://github.com/kallebylin/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/kallebylin/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/kallebylin/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/kallebylin/repo.svg?style=for-the-badge
[forks-url]: https://github.com/kallebylin/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/kallebylin/repo.svg?style=for-the-badge
[stars-url]: https://github.com/kallebylin/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/kallebylin/repo.svg?style=for-the-badge
[issues-url]: https://github.com/kallebylin/repo_name/issues
[license-shield]: https://img.shields.io/github/license/kallebylin/repo.svg?style=for-the-badge
[license-url]: https://github.com/kallebylin/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kallebylin
