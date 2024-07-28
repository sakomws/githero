WIP


Start db:
```
brew install postgresql
brew services start postgresql
brew services list
initdb /usr/local/var/postgres
psql postgres
CREATE ROLE postgres  WITH LOGIN;
ALTER ROLE postgres  WITH SUPERUSER;
```


lsof -i :5432kill -9 