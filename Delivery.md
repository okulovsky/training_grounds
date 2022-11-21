## Как создать контейнер

1. Из корня проекта, в котором используется ```tg```, запустить скрипт ```dependencies_fix.py```

2. Сделать инстанс класса ```SSHDockerJobRoutine```, передав в конструктор необходимую работу, например, как в демке [DeliverableJobs](https://github.com/okulovsky/training_grounds/blob/v2/demos/4.2.%20Deliverable%20Jobs%20(tg.common.delivery.jobs).ipynb).

3. Убедиться, что докер выполняет команды без ```sudo```. Если это не так, то [выполнить шаги](https://docs.docker.com/engine/install/linux-postinstall/) из документации.

4. Запустить сборку контейнера с помощью метода ```build_container``` из ```tg/common/delivery/jobs/ssh_docker_job_routine```.

В результате по команде ```docker images``` должен отображаться образ с именем и тэгом, указанными при создании на последнем шаге. *Ниже будет использоваться имя ```local-image:tagname```*.

## Как доставить контейнер на сервер

1. Если нет аккаунта на [DockerHub](https://hub.docker.com/), то зарегистрироваться и создать публичный репозиторий, например, ```new-repo```.

*Дальнейшие шаги выполнять через CLI*

2. Залогиниться с помощью команды ```docker login --username=user```, ввести пароль от аккаунта.

3. Поставить тэг на локальный репозиторий, который должен быть отправлен на хаб ```docker tag local-image:tagname new-repo:new-tagname```

4. Запушить в репозиторий командой ```docker push new-repo:tagname```

## Как запустить выполнение

В ячейке ноутбука ```DataSphere``` выполнить команду ```#!:docker-run <Имя образа>```. Изменить вычислительные ресурсы ячейки можно префиксом, как в [документации](https://cloud.yandex.ru/docs/datasphere/operations/user-images-cell)