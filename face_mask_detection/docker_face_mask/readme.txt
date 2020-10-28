cd in to docker_face_mask folder
run docker build -t docker_face_mask .t

wait for image to build, has tensorflow etc so will take its time

docker run -d -p 5000:5000 docker_face_mask

runs the containers with your image in background in detached mode
if any error it may exit immediately so replace -d with -it for interactive mode to see output of running the file to see if errors exist,
also allows us to execute commands at the time of running the containter.
 - my understanding of docker began yday so everything written here could be garbage

delete images
 - docker images: to list all images
 - docker rmi  [image id] to remove image by id
 
docker containers
 - docker ps: to see all running containers
 - docker ps -a: to see all containers that have ever run
 - docker stop [container name] stop running container
 - docker rm [container id] to remove container by id