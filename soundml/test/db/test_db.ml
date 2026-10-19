open Windtrap

let () =
  exit
    (run "Db"
       ( Test_golden.suite @ Test_props.suite @ Test_stage.suite
       @ Accept_value_stage.suite ) )
