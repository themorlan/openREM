FROM osimis/orthanc
RUN apt-get update && apt-get -y install  zip unzip
COPY openrem_orthanc_config_docker.lua /etc/share/orthanc/scripts/
#RUN sed -i '/LuaScripts.*/a \ \ \ \ \"/etc/share/orthanc/scripts/openrem_orthanc_config_docker.lua\"' /etc/orthanc/orthanc.json