import tritonclient.grpc as grpcclient

class Singleton(object):
    _instance = None

    def __new__(
            class_,
            verbose=False,
            url="localhost:8001",
            grpc_keepalive_time=2 ** 31 - 1,
            grpc_keepalive_timeout=20000,
            grpc_keepalive_permit_without_calls=False,
            grpc_http2_max_pings_without_data=2,
    ):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_)
        return class_._instance


class TritonClient(Singleton):
    def __init__(self,
                 verbose=False,
                 url="localhost:8001",
                 grpc_keepalive_time=2**31-1,
                 grpc_keepalive_timeout=20000,
                 grpc_keepalive_permit_without_calls=False,
                 grpc_http2_max_pings_without_data=2,
                 ):
        self.verbose = verbose
        self.url = url
        self.grpc_keepalive_time = grpc_keepalive_time
        self.grpc_keepalive_timeout = grpc_keepalive_timeout
        self.grpc_keepalive_permit_without_calls = grpc_keepalive_permit_without_calls
        self.grpc_http2_max_pings_without_data = grpc_http2_max_pings_without_data

        try:
            keepalive_options = grpcclient.KeepAliveOptions(
                keepalive_time_ms=self.grpc_keepalive_time,
                keepalive_timeout_ms=self.grpc_keepalive_timeout,
                keepalive_permit_without_calls=self.grpc_keepalive_permit_without_calls,
                http2_max_pings_without_data=self.grpc_http2_max_pings_without_data
            )
            self.triton_client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=self.verbose,
                keepalive_options=keepalive_options
            )
        except Exception as e:
            raise Exception("channel creation failed: " + str(e))


if __name__ == '__main__':
    tritonclient1 = TritonClient(url='localhost:8001')
    tritonclient2 = TritonClient()
    print(tritonclient2 is tritonclient1)