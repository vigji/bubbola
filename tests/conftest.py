def pytest_addoption(parser):
    parser.addoption(
        "--run-model-tests",
        action="store_true",
        default=False,
        help="run tests that require actual model/API access",
    )
