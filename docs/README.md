# Docs

To generate the API reference, run the following command inside the `docs` directory.

```
rm -rf source
sphinx-apidoc -f -e -o source/api ../src/abcmodel
make html
```
