# DisableDockerDetector "Disabled to unblock PRs until the owner can fix the file. Not used in any prod deployments - only as a documentation for the customers"
FROM rocker/tidyverse:4.0.0-ubuntu20.04
 
# Install python
RUN apt-get update -qq && \
 apt-get install -y python3
 
# Create link for python
RUN ln -f /usr/bin/python3 /usr/bin/python

# Install additional R packages
RUN R -e "install.packages(c('optparse'), repos = 'https://cloud.r-project.org/')"