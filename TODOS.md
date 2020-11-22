# Learning objectives

# The "what's"
- What is Docker? And Docker Compose?
- What is VSCode?

# The "why's"
- Why should I use Docker containers to encapsulate my Deep Learning projects?
    - no different CUDA versions trashing your system
    - can run on different platforms => portability & reusability
    - deploy the container to cloud providers (AWS, GCloud, Azure etc.)
- Why should I care about debugging the DL containers?
    - Sometimes stuff goes wrong
- Why should I use Docker compose? 
    - TBD :)

# The "how's"
- How do I build a Docker container with all my code in it?
- How & why do I want to load the MNIST data set during Docker build?
    - Docker containers use a writable layer to store files created in the container. 
    This means that if we download the MNIST data set during runtime of our container, the files 
    will not be persisted after the container no longer exists. As a result, the data set will be 
    downloaded every time we restart the container, which is not good practice. 
    see https://docs.docker.com/storage/ 
- How do I debug the code when running the container?
    - Mount the src dir into the container during development, but NEVER for production!
- How do I save my model to the host PC?
    - Mount another dir and save to that, easy-peasy!
- How do I save logs on the host PC? And why?
    - Why: because then they persist after the container is removed

