def rule_based_decision(applicant):
    income = applicant.get('ApplicantIncome', 0)
    loan = applicant.get('LoanAmount', 0)
    credit = applicant.get('Credit_History', 0)

    if credit == 1 and income > 40000:
        return "Approve"
    elif credit == 0:
        return "Reject"
    elif loan > income * 0.6:
        return "Risky"
    else:
        return "Review"