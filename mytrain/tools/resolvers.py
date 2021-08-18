from class_resolver import Resolver


def add_resolver(resolver, cls):
    """
    添加自定义类型到类型查找器中
    """
    resolver.lookup_dict[resolver.normalize_cls(cls)] = cls


model_resolver = Resolver(
    {},
    base=object,  # type: ignore
    default=None,
    suffix='',
)

optimizer_resolver = Resolver(
    {},
    base=object,  # type: ignore
    default=None,
    suffix='',
)

dataset_resolver = Resolver(
    {},
    base=object,  # type: ignore
    default=None,
    suffix='Dataset',
)

if __name__ == '__main__':
    assert dataset_resolver.lookup("SRNSML1M") == SRNSML1MDataset
    assert dataset_resolver.lookup("NCFPinterest") == NCFPinterestDataset
