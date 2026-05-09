def rank_applicants(applicants):
    def score(app):
        income = app.get('ApplicantIncome', 0)
        credit = app.get('Credit_History', 0)
        loan = app.get('LoanAmount', 0)

        return (income * 0.5) + (credit * 10000) - (loan * 10)

    return sorted(applicants, key=score, reverse=True)