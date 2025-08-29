from abc import abstractmethod


class AbstractLandSurfaceModel:
    def __init__(self):
        pass

    @abstractmethod
    def run(
        self,
    ) -> None:
        raise NotImplementedError

    def integrate(
        self,
    ) -> None:
        pass


class NoLandSurfaceModel(AbstractLandSurfaceModel):
    def __init__(self):
        super().__init__()

    def run(
        self,
    ) -> None:
        pass


class JarvisStewartModel(AbstractLandSurfaceModel):
    def __init__(self):
        super().__init__()

    def run(
        self,
    ) -> None:
        pass


class AGSModel(AbstractLandSurfaceModel):
    def __init__(self):
        super().__init__()

    def run(
        self,
    ) -> None:
        pass
