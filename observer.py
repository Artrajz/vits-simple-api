from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class Subject(ABC):
    def __init__(self):
        self._observers = set()

    def attach(self, observer: Observer):
        self._observers.add(observer)

    def detach(self, observer: Observer):
        self._observers.discard(observer)

    def notify(self, *args, **kwargs):
        for observer in self._observers:
            observer.update(*args, **kwargs)
