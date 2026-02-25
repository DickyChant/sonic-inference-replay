import json
import os
import time
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.dump_dir = "/dumps"
        self.device = torch.device("cuda:0")

        # Manual switch: set SONIC_DUMP_INPUTS=1 to dump every request's inputs
        self.always_dump = os.environ.get("SONIC_DUMP_INPUTS", "0") == "1"
        if self.always_dump:
            print(f"[SONICDebug] SONIC_DUMP_INPUTS enabled — dumping all inputs to {self.dump_dir}")
        self.dump_counter = 0

        self.model = torch.jit.load("/models/particlenet_AK4_PT/1/model.pt", map_location=self.device).to(self.device)
        self.model.eval()

        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(model_config, "softmax__0")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def _dump_inputs(self, request_inputs, tag="manual"):
        """Save request inputs as .npz for offline replay."""
        self.dump_counter += 1
        filename = f"sonic_req_{tag}_{int(time.time())}_{self.dump_counter}.npz"
        path = os.path.join(self.dump_dir, filename)
        np.savez(path, **request_inputs)
        print(f"[SONICDebug] Dumped inputs to {path}")

    def execute(self, requests):
        responses = []

        with torch.no_grad():
            for request in requests:
                request_inputs = {
                    "pf_points__0": pb_utils.get_input_tensor_by_name(request, "pf_points__0").as_numpy(),
                    "pf_features__1": pb_utils.get_input_tensor_by_name(request, "pf_features__1").as_numpy(),
                    "pf_mask__2": pb_utils.get_input_tensor_by_name(request, "pf_mask__2").as_numpy(),
                    "sv_points__3": pb_utils.get_input_tensor_by_name(request, "sv_points__3").as_numpy(),
                    "sv_features__4": pb_utils.get_input_tensor_by_name(request, "sv_features__4").as_numpy(),
                    "sv_mask__5": pb_utils.get_input_tensor_by_name(request, "sv_mask__5").as_numpy()
                }

                # Manual trigger: dump every request when switch is on
                if self.always_dump:
                    self._dump_inputs(request_inputs, tag="manual")

                try:
                    pf_points = torch.from_numpy(request_inputs["pf_points__0"]).to(self.device)
                    pf_features = torch.from_numpy(request_inputs["pf_features__1"]).to(self.device)
                    pf_mask = torch.from_numpy(request_inputs["pf_mask__2"]).to(self.device)
                    sv_points = torch.from_numpy(request_inputs["sv_points__3"]).to(self.device)
                    sv_features = torch.from_numpy(request_inputs["sv_features__4"]).to(self.device)
                    sv_mask = torch.from_numpy(request_inputs["sv_mask__5"]).to(self.device)

                    output_tensor = self.model(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)

                    output = pb_utils.Tensor("softmax__0", output_tensor.cpu().numpy().astype(self.output_dtype))
                    response = pb_utils.InferenceResponse(output_tensors=[output])

                except Exception as ex:
                    err_msg = f"Error during inference: {str(ex)}"
                    response = pb_utils.InferenceResponse(error=pb_utils.TritonError(err_msg))
                    self._dump_inputs(request_inputs, tag="error")

                responses.append(response)

        return responses
