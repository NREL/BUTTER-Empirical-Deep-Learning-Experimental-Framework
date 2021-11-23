# Notes for PGPool installation.

Host: screech.hpc.nrel.gov
- 32 CPUS
- 10GB networking to Eagle
- 263Gb RAM
- To manage docker on screech, you have to be in the dockerroot group.
- Andy Archer has been my POC to get this installed and running

## Set up bitnami image

Pull image from dockerhub
```
docker pull bitnami/pgpool:4.2.6
```

Set up sensitive environment variables
```
export DMP_DB_USER=dmpappsops
export DMP_DB_PASSWORD=
export PGPOOL_ADMIN_USERNAME= # You can make this up
export PGPOOL_ADMIN_PASSWORD= # You can make this up
```

Run the container, binding to port 5432
More information about configuration parameters can be found on Github: https://github.com/bitnami/bitnami-docker-pgpool
```
docker run -p 5432:5432 --detach --rm --name pgpool \
  --env PGPOOL_BACKEND_NODES=0:yuma.hpc.nrel.gov:5432 \
  --env PGPOOL_SR_CHECK_USER=$DMP_DB_USER \
  --env PGPOOL_SR_CHECK_PASSWORD=$DMP_DB_PASSWORD \
  --env PGPOOL_ENABLE_LDAP=no \
  --env PGPOOL_POSTGRES_USERNAME=$DMP_DB_USER \
  --env PGPOOL_POSTGRES_PASSWORD=$DMP_DB_PASSWORD \
  --env PGPOOL_ADMIN_USERNAME=$PGPOOL_ADMIN_USERNAME \
  --env PGPOOL_ADMIN_PASSWORD=$PGPOOL_ADMIN_PASSWORD \
  bitnami/pgpool:4.2.6
```

## Managing a running container

```
docker container list
docker exec -it pgpool /bin/bash
```