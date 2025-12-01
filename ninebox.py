# ninebox.py

class NineBox:

    @staticmethod
    def get_nine_box(performance, potential):
        if performance >= 4 and potential >= 4:
            return ("High Performer", "High Potential", "Stars")
        elif performance >= 4 and potential == 3:
            return ("High Performer", "Medium Potential", "Consistent Stars")
        elif performance == 3 and potential >= 4:
            return ("Medium Performer", "High Potential", "High Potentials")
        elif performance == 3 and potential == 3:
            return ("Medium Performer", "Medium Potential", "Core Players")
        elif performance <= 2 and potential >= 4:
            return ("Low Performer", "High Potential", "Emerging")
        elif performance >= 4 and potential <= 2:
            return ("High Performer", "Low Potential", "Workhorses")
        elif performance == 3 and potential <= 2:
            return ("Medium Performer", "Low Potential", "Underperformer")
        elif performance <= 2 and potential == 3:
            return ("Low Performer", "Medium Potential", "Inconsistent")
        else:
            return ("Low Performer", "Low Potential", "Risk Zone")

    @staticmethod
    def apply(data):
        """
        data = list of employee objects returned from AI
        """

        for emp in data:
            q = emp["employee_analysis"]["quantitative_scores"]

            perf = q["performance_score"]
            pot = q["potential_score"]

            p_label, pot_label, grid_label = NineBox.get_nine_box(perf, pot)

            emp["employee_analysis"]["nine_box_performance"] = p_label
            emp["employee_analysis"]["nine_box_potential"] = pot_label
            emp["employee_analysis"]["nine_box_label"] = grid_label

        return data
