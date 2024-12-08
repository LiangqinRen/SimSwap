mode=$1
mode=`echo $mode | tr '[:upper:]' '[:lower:]'`

if [ $# == 2 ] && [ "$2" = "console" ]; then
    console_only="--console_only"
else
    console_only=""
fi

if [[ $mode == 'pgd_both_sample' ]] || [[ $mode == 'pgd_both_robustness_sample' ]]
then
    python main.py --method $mode --epochs 300 --pgd_epsilon 0.005 --pgd_limit 0.075 $console_only
elif [[ $mode == 'pgd_both_metric' ]] || [[ $mode == 'pgd_both_robustness_metric' ]]
then
    python main.py --method $mode --epochs 300 --pgd_epsilon 0.005 --pgd_limit 0.075 --batch_size 12 $console_only
elif [[ $mode == 'gan_both_train' ]]
then
    python main.py --method $mode --batch_size 7 --log_interval 250 --epochs 500000 $console_only
elif [[ $mode == 'gan_both_sample' ]] || [[ $mode == 'gan_both_robustness_sample' ]]
then
    python main.py --method $mode --gan_test_models "gan_both.pth" $console_only
elif [[ $mode == 'gan_both_metric' ]] || [[ $mode == 'gan_both_robustness_metric' ]]
then
    python main.py --method $mode --gan_test_models "gan_both.pth" --batch_size 7 $console_only
elif [[ $mode == 'worker' ]]
then
    python main.py --method $mode --batch_size 64 $console_only
elif [[ $mode == 'distance' ]]
then
    python main.py --method $mode --batch_size 64 $console_only
elif [[ $mode == 'robustness' ]]
then
    python main.py --method $mode --batch_size 14 $console_only
elif [[ $mode == 'diff_identity_match' ]]
then
    python main.py --method $mode --batch_size 14 --anchor_dir "fake" $console_only
else
    echo "Unrecognized mode!"
fi