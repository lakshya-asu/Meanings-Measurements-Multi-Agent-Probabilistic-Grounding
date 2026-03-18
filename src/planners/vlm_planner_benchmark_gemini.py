import json
from enum import Enum
import time
import base64

import google.generativeai as genai
import os
import mimetypes
from src.utils.data_utils import get_latest_image

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Choose a Gemini model.
gemini_model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")  # "models/gemini-2.5-pro-preview-03-25", etc.


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ============================================================
# ORIGINAL EQA RESPONSE SCHEMA (unchanged)
# ============================================================
def create_planner_response(frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options):
    frontier_step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "explanation_frontier": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain reasoning for choosing this frontier to explore by referencing list of objects (<id> and <name>) connected to that frontier node via a link (refer to scene graph).",
            ),
            "frontier_id": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.value for member in frontier_node_list],
            ),
        },
        required=["explanation_frontier", "frontier_id"],
    )

    object_step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "explanation_room": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain very briefly reasoning for selecting this room.",
            ),
            "room_id": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.name for member in room_node_list],
                description="Choose the room which contains the object you want to goto.",
            ),
            "room_name": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Refer to the the scene graph to output the room_name corresponding to the selected room_id",
            ),
            "explanation_obj": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain very briefly reasoning for selecting this object in the selected room.",
            ),
            "object_id": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[member.name for member in object_node_list],
                description="Only select from objects within the room chosen.",
            ),
            "object_name": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Refer to the the scene graph to output the object_name corresponding to the selected object_id",
            ),
        },
        required=["explanation_room", "explanation_obj", "room_id", "room_name", "object_id", "object_name"],
    )

    answer = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "explanation_ans": genai.protos.Schema(type=genai.protos.Type.STRING, description="Select the correct answer from the options."),
            "answer": genai.protos.Schema(type=genai.protos.Type.STRING, enum=[member.name for member in Answer_options]),
            "value": genai.protos.Schema(type=genai.protos.Type.STRING, enum=[member.value for member in Answer_options]),
            "explanation_conf": genai.protos.Schema(type=genai.protos.Type.STRING, description="Explain the reasoning behind the confidence level of your answer."),
            "confidence_level": genai.protos.Schema(type=genai.protos.Type.NUMBER, description="Rate confidence 0..1."),
            "is_confident": genai.protos.Schema(
                type=genai.protos.Type.BOOLEAN,
                description="TRUE only with visual confirmation + SG grounding. FALSE if uncertain.",
            ),
        },
        required=["explanation_ans", "answer", "value", "explanation_conf", "confidence_level", "is_confident"],
    )

    image_description = genai.protos.Schema(
        type=genai.protos.Type.STRING,
        description="Describe the CURRENT IMAGE. Pay special attention to features that can help answer the question or select future actions.",
    )
    scene_graph_description = genai.protos.Schema(
        type=genai.protos.Type.STRING,
        description="Describe the SCENE GRAPH. Pay special attention to features that can help answer the question or select future actions.",
    )
    question_type = genai.protos.Schema(
        type=genai.protos.Type.STRING,
        enum=["Identification", "Counting", "Existence", "State", "Location"],
        description="Type of question.",
    )

    step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={"Goto_frontier_node_step": frontier_step, "Goto_object_node_step": object_step},
        description="Choose only one of 'Goto_frontier_node_step', 'Goto_object_node_step'.",
    )
    steps = genai.protos.Schema(type=genai.protos.Type.ARRAY, items=step, min_items=1)

    response_schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "steps": steps,
            "image_description": image_description,
            "scene_graph_description": scene_graph_description,
            "question_type": question_type,
            "answer": answer,
        },
        required=["steps", "image_description", "scene_graph_description", "question_type", "answer"],
    )
    return response_schema

###
# ============================================================
# NEW MSP RESPONSE SCHEMAS (OBJECT mode + POINT mode)
# ============================================================
def _common_steps_schema(frontier_node_list, room_node_list, region_node_list, object_node_list):
    frontier_step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "explanation_frontier": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain reasoning for choosing this frontier to explore by referencing objects (<id> and <name>) connected to that frontier node.",
            ),
            "frontier_id": genai.protos.Schema(type=genai.protos.Type.STRING, enum=[m.value for m in frontier_node_list]),
        },
        required=["explanation_frontier", "frontier_id"],
    )

    object_step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "explanation_room": genai.protos.Schema(type=genai.protos.Type.STRING, description="Very brief reasoning for selecting this room."),
            "room_id": genai.protos.Schema(type=genai.protos.Type.STRING, enum=[m.name for m in room_node_list]),
            "room_name": genai.protos.Schema(type=genai.protos.Type.STRING, description="Room name corresponding to room_id."),
            "explanation_obj": genai.protos.Schema(type=genai.protos.Type.STRING, description="Very brief reasoning for selecting this object."),
            "object_id": genai.protos.Schema(type=genai.protos.Type.STRING, enum=[m.name for m in object_node_list]),
            "object_name": genai.protos.Schema(type=genai.protos.Type.STRING, description="Object name corresponding to object_id."),
        },
        required=["explanation_room", "explanation_obj", "room_id", "room_name", "object_id", "object_name"],
    )

    image_description = genai.protos.Schema(
        type=genai.protos.Type.STRING,
        description="Describe the CURRENT IMAGE. Focus on cues that help answer MSP question or select actions.",
    )
    scene_graph_description = genai.protos.Schema(
        type=genai.protos.Type.STRING,
        description="Describe the SCENE GRAPH. Focus on cues that help answer MSP question or select actions.",
    )
    question_type = genai.protos.Schema(
        type=genai.protos.Type.STRING,
        enum=["Identification", "Counting", "Existence", "State", "Location"],
        description="Type of question.",
    )

    step = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={"Goto_frontier_node_step": frontier_step, "Goto_object_node_step": object_step},
        description="Choose only one of 'Goto_frontier_node_step', 'Goto_object_node_step'.",
    )
    steps = genai.protos.Schema(type=genai.protos.Type.ARRAY, items=step, min_items=1)
    return steps, image_description, scene_graph_description, question_type


def create_planner_response_msp_object(frontier_node_list, room_node_list, region_node_list, object_node_list):
    steps, image_description, scene_graph_description, question_type = _common_steps_schema(
        frontier_node_list, room_node_list, region_node_list, object_node_list
    )

    answer = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "explanation_ans": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain why this selected object best corresponds to the MSP target location.",
            ),
            "anchor_object_id": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Anchor object id from the MSP question (e.g., Plant_526).",
            ),
            "selected_object_id": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=[m.name for m in object_node_list],
                description="Pick the object whose center best matches the MSP query target location.",
            ),
            "selected_object_name": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Object name corresponding to selected_object_id.",
            ),
            "explanation_conf": genai.protos.Schema(type=genai.protos.Type.STRING, description="Explain confidence."),
            "confidence_level": genai.protos.Schema(type=genai.protos.Type.NUMBER, description="0..1"),
            "is_confident": genai.protos.Schema(
                type=genai.protos.Type.BOOLEAN,
                description="TRUE only with visual confirmation + SG grounding. FALSE if uncertain.",
            ),
        },
        required=[
            "explanation_ans",
            "anchor_object_id",
            "selected_object_id",
            "selected_object_name",
            "explanation_conf",
            "confidence_level",
            "is_confident",
        ],
    )

    response_schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "steps": steps,
            "image_description": image_description,
            "scene_graph_description": scene_graph_description,
            "question_type": question_type,
            "answer": answer,
        },
        required=["steps", "image_description", "scene_graph_description", "question_type", "answer"],
    )
    return response_schema


def create_planner_response_msp_point(frontier_node_list, room_node_list, region_node_list, object_node_list):
    steps, image_description, scene_graph_description, question_type = _common_steps_schema(
        frontier_node_list, room_node_list, region_node_list, object_node_list
    )

    answer = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "explanation_ans": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Explain why this [x,y,z] corresponds to the MSP target location.",
            ),
            "anchor_object_id": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="Anchor object id from the MSP question (e.g., Plant_526).",
            ),
            "target_point_xyz": genai.protos.Schema(
                type=genai.protos.Type.ARRAY,
                items=genai.protos.Schema(type=genai.protos.Type.NUMBER),
                description="Return [x, y, z] in HABITAT WORLD coordinates (meters).",
            ),
            "explanation_conf": genai.protos.Schema(type=genai.protos.Type.STRING, description="Explain confidence."),
            "confidence_level": genai.protos.Schema(type=genai.protos.Type.NUMBER, description="0..1"),
            "is_confident": genai.protos.Schema(
                type=genai.protos.Type.BOOLEAN,
                description="TRUE only with visual confirmation + SG grounding. FALSE if uncertain.",
            ),
        },
        required=["explanation_ans", "anchor_object_id", "target_point_xyz", "explanation_conf", "confidence_level", "is_confident"],
    )

    response_schema = genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "steps": steps,
            "image_description": image_description,
            "scene_graph_description": scene_graph_description,
            "question_type": question_type,
            "answer": answer,
        },
        required=["steps", "image_description", "scene_graph_description", "question_type", "answer"],
    )
    return response_schema


# ============================================================
# PLANNER CLASS (EQA + MSP with minimal branching)
# ============================================================
class VLMPlannerEQAGemini:
    """
    Backwards-compatible:
      - EQA usage: same as before (pred_candidates, choices, answer)
      - MSP usage: pass mode in cfg (cfg.answer_mode), and pass anchor_object_id
    """

    def __init__(
        self,
        cfg,
        sg_sim,
        question,
        pred_candidates=None,
        choices=None,
        answer=None,
        output_path=None,
        anchor_object_id=None,
    ):
        self._question = question
        self.choices = choices if choices is not None else []
        self.vlm_pred_candidates = pred_candidates if pred_candidates is not None else []
        self._answer = answer
        self._output_path = output_path
        self._vlm_type = cfg.name
        self._use_image = cfg.use_image

        # MSP mode: "eqa" (default), "msp_object", "msp_point"
        self._answer_mode = getattr(cfg, "answer_mode", "eqa")
        self._anchor_object_id = anchor_object_id

        self._example_plan = ""
        self._history = ""
        self.full_plan = ""
        self._t = 0
        self._add_history = cfg.add_history

        self._outputs_to_save = [f"Question: {self._question}. \n Answer: {self._answer} \n"]
        self.sg_sim = sg_sim

    @property
    def t(self):
        return self._t

    def get_actions(self):
        object_node_list = Enum(
            "object_node_list",
            {id: name for id, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names)},
            type=str,
        )

        if len(self.sg_sim.frontier_node_ids) > 0:
            frontier_node_list = Enum("frontier_node_list", {ac: ac for ac in self.sg_sim.frontier_node_ids}, type=str)
        else:
            frontier_node_list = Enum("frontier_node_list", {"frontier_0": "Do not choose this option. No more frontiers left."}, type=str)

        room_node_list = Enum(
            "room_node_list",
            {id: name for id, name in zip(self.sg_sim.room_node_ids, self.sg_sim.room_node_names)},
            type=str,
        )
        region_node_list = Enum("region_node_list", {ac: ac for ac in self.sg_sim.region_node_ids}, type=str)

        if self._answer_mode == "eqa":
            Answer_options = Enum(
                "Answer_options",
                {token: choice for token, choice in zip(self.vlm_pred_candidates, self.choices)},
                type=str,
            )
            return frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options

        # MSP modes do NOT use Answer_options
        return frontier_node_list, room_node_list, region_node_list, object_node_list, None

    @property
    def agent_role_prompt(self):
        scene_graph_desc = (
            "A scene graph represents an indoor environment in a hierarchical tree structure consisting of nodes and edges/links. "
            "There are six types of nodes: building, rooms, visited areas, frontiers, objects, and agent in the environment.\n"
            "The tree structure is as follows: At the highest level 5 is a 'building' node.\n"
            "At level 4 are room nodes. There are links connecting the building node to each room node.\n"
            "At level 3 are region and frontier nodes. Region nodes represent explored areas; frontier nodes represent boundaries of explored/unexplored.\n"
            "There are links from room nodes to region/frontier nodes indicating which room they belong to.\n"
            "At level 2 are object nodes and agent nodes. There is an edge from region node to each object node depicting which visited area the object is in.\n"
            "There are also links between frontier nodes and object nodes depicting objects in the vicinity of a frontier.\n"
            "Finally the agent node is where you are located; there is an edge between region and agent depicting which visited area the agent is in.\n"
        )

        current_state_des = "'CURRENT STATE' will give you the exact location of the agent in the scene graph by giving you the agent node id, location, room_id and room name."
        if self._use_image:
            current_state_des += " Additionally, you will also be given the current view of the agent as an image."

        # NOTE: no direction maps; just output format for MSP point answers
        msp_extra = ""
        if self._answer_mode == "msp_point":
            msp_extra = (
                "\nIf answering with a point, output [x,y,z] as HABITAT WORLD coordinates in meters, consistent with object centers in the scene graph."
            )

        prompt = f"""You are an excellent hierarchical graph planning agent.
Your goal is to navigate an unseen environment to confidently answer a question about the environment.
As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.
{scene_graph_desc}
{current_state_des}

Given the current state information, try to answer the question. Explain the reasoning for your answer.
Finally, report whether you are confident in answering the question.
Explain the reasoning behind the confidence level of your answer. Rate your level of confidence between 0 and 1.

Do not use just commonsense knowledge to decide confidence.
Choose TRUE only when you have a visual confirmation (from the image if provided) as well as from the scene graph that your answer is correct.
Choose FALSE if you are uncertain and should explore more to ground your answer in the environment.

If you are unable to answer with high confidence, you can take two kinds of steps:
- Goto_object_node_step: navigate near a certain object in the scene graph.
- Goto_frontier_node_step: navigate to a frontier (unexplored region) to expand the scene graph.

While choosing actions, pay close attention to HISTORY and avoid repeating actions.
Describe the CURRENT IMAGE (if provided) and the SCENE GRAPH.

{msp_extra}
"""
        prompt_no_image = prompt.replace("Describe the CURRENT IMAGE (if provided) and the SCENE GRAPH.", "Describe the SCENE GRAPH.")

        return prompt if self._use_image else prompt_no_image

    def get_current_state_prompt(self, scene_graph, agent_state):
        prompt = f"At t = {self.t}:\nCURRENT AGENT STATE: {agent_state}.\nSCENE GRAPH: {scene_graph}.\n"
        if self._add_history:
            prompt += f"HISTORY: {self._history}"
        return prompt

    def update_history(self, agent_state, step, question_type):
        if step["step_type"] == "Goto_object_node_step":
            action = f"Goto object_id: {step['choice']} object name: {step['value']}"
        elif step["step_type"] == "Goto_frontier_node_step":
            action = f"Goto frontier_id: {step['choice']}"
        else:
            action = f"Answer: {step.get('choice')}"

        last_step = f"""
[Agent state(t={self.t}): {agent_state},
Action(t={self.t}): {action},
Question Type: {question_type}]
"""
        self._history += last_step

    def _build_messages(self, current_state_prompt):
        messages = [
            {"role": "model", "parts": [{"text": f"AGENT ROLE: {self.agent_role_prompt}"}]},
            {"role": "model", "parts": [{"text": f"QUESTION: {self._question}"}]},
            {"role": "user", "parts": [{"text": f"CURRENT STATE: {current_state_prompt}."}]},
        ]

        if self._use_image:
            image_path = get_latest_image(self._output_path)
            base64_image = encode_image(image_path)
            mime_type = mimetypes.guess_type(image_path)[0]
            messages.append(
                {
                    "role": "user",
                    "parts": [
                        {"text": "CURRENT IMAGE: This image represents the current view of the agent. Use this as additional information."},
                        {"inline_data": {"mime_type": mime_type, "data": base64_image}},
                    ],
                }
            )
        return messages

    def get_gemini_output(self, current_state_prompt):
        messages = self._build_messages(current_state_prompt)

        frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options = self.get_actions()

        if self._answer_mode == "eqa":
            response_schema = create_planner_response(frontier_node_list, room_node_list, region_node_list, object_node_list, Answer_options)
        elif self._answer_mode == "msp_object":
            response_schema = create_planner_response_msp_object(frontier_node_list, room_node_list, region_node_list, object_node_list)
        elif self._answer_mode == "msp_point":
            response_schema = create_planner_response_msp_point(frontier_node_list, room_node_list, region_node_list, object_node_list)
        else:
            raise ValueError(f"Unknown answer_mode: {self._answer_mode}")

        succ = False
        while not succ:
            try:
                start = time.time()
                response = gemini_model.generate_content(
                    messages,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                        response_schema=response_schema,
                    ),
                )
                print(f"Time taken for planning next step: {time.time() - start}s")
                succ = True
            except Exception as e:
                print(f"An error occurred: {e}. Sleeping for 45s")
                time.sleep(45)

        response_dict = json.loads(response.text)

        step_out = response_dict["steps"][0]
        sg_desc = response_dict["scene_graph_description"]
        img_desc = response_dict["image_description"] if self._use_image else " "
        question_type = response_dict["question_type"]
        answer = response_dict["answer"]

        # Force anchor id into answer if provided (doesn't change reasoning; just ensures it is always present for logging)
        if self._answer_mode.startswith("msp") and self._anchor_object_id is not None:
            answer["anchor_object_id"] = self._anchor_object_id

        # Parse step (unchanged)
        if step_out:
            step = {}
            step_type = list(step_out.keys())[0]
            step["step_type"] = step_type
            if step_type == "Goto_object_node_step":
                step["choice"] = step_out[step_type]["object_id"]
                step["value"] = step_out[step_type]["object_name"]
                step["explanation"] = step_out[step_type]["explanation_obj"]
                step["room"] = step_out[step_type]["room_name"]
                step["explanation_room"] = step_out[step_type]["explanation_room"]
            elif step_type == "Goto_frontier_node_step":
                step["choice"] = step_out[step_type]["frontier_id"]
                step["explanation"] = step_out[step_type]["explanation_frontier"]
            else:
                step = None
        else:
            step = None

        return step, img_desc, sg_desc, question_type, answer

    def get_next_action(self):
        agent_state = self.sg_sim.get_current_semantic_state_str()
        current_state_prompt = self.get_current_state_prompt(self.sg_sim.scene_graph_str, agent_state)

        step, img_desc, sg_desc, question_type, answer = self.get_gemini_output(current_state_prompt)

        print(f"At t={self._t}:\n Step: {step}\n Answer: {answer}\n Question type: {question_type}")

        # Save outputs
        self._outputs_to_save.append(
            f"At t={self._t}:\n"
            f"Agent state: {agent_state}\n"
            f"VLM step: {step}\n"
            f"Image desc: {img_desc}\n"
            f"Scene graph desc: {sg_desc}\n"
            f"Question type: {question_type}\n"
            f"Answer: {answer}\n"
        )
        self.full_plan = " ".join(self._outputs_to_save)
        with open(self._output_path / "llm_outputs.json", "w") as text_file:
            text_file.write(self.full_plan)

        # If no step / no frontiers, stop
        if step is None or step.get("choice") == "Do not choose this option. No more frontiers left.":
            return None, None, answer.get("is_confident", False), answer.get("confidence_level", 0.0), answer

        if self._add_history:
            self.update_history(agent_state, step, question_type)

        self._t += 1

        target_pose = self.sg_sim.get_position_from_id(step["choice"])
        target_id = step["choice"]

        # IMPORTANT: return the full answer dict (runner will interpret depending on mode)
        return target_pose, target_id, answer.get("is_confident", False), answer.get("confidence_level", 0.0), answer
