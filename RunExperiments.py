from PysparkMSTedgesampling import main

# To count at which configuration we are
i = 1

for perc_leave_out in [0, 20, 40, 60, 80]:
    for between_clusters in [False, True]:
        main(perc_leave_out, between_clusters, i)
        i += 1